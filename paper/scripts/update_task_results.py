#!/usr/bin/env python3
"""Replace complete tasks in a historical result with validated targeted retests.

The output is explicitly a combined record set with explicit controller provenance,
not a fresh 400-episode evaluation of the updated controller.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

PAPER = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def project(row, line, campaign, source):
    trace = row['route_trace']
    return {
        'episode_id': row['episode_id'], 'suite': row['key']['suite'],
        'task_id': row['key']['task_id'], 'init_id': row['key']['episode_index'],
        **{key: row[key] for key in ('instruction', 'seed', 'policy_status', 'failure', 'steps', 'elapsed_s')},
        'success': row['evaluator_success'], 'reused': False,
        'final_phase': trace.get('phase'),
        'grasp_attempts': len(trace.get('grasp_target_attempts', [])),
        'grasp_checks': len(trace.get('grasp_verifications', [])),
        'rejected_grasp_checks': sum(not item.get('accepted', False) for item in trace.get('grasp_verifications', [])),
        'placement_attempts': len(trace.get('placement_target_attempts', [])),
        'scoring': trace['evaluator_isolation']['scoring'],
        'source_record_sha256': sha(line), 'source_campaign': campaign,
        'controller_source_sha256': source,
    }


CAPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def validate_task(task_root):
    manifest = read(task_root/'manifest.json')
    status = read(task_root/'status.json')
    assert status['state']=='completed', 'The targeted retest is not complete.'
    validation = read(task_root/'final-validation.json')
    assert validation['coverage_passed'] and not validation['issues']
    assert len(manifest['tasks'])==1
    task = manifest['tasks'][0]
    assert task['suite'] in CAPS and 0 <= task['task_id'] < 10
    protocol = manifest['protocol']
    assert protocol['official_init_ids']==list(range(10))
    assert protocol['seed']==7 and protocol['episodes_per_task']==10
    assert protocol['new_episodes']==10 and protocol['reused_episodes']==0
    assert protocol['scoring']=='external_sticky_any_success'
    assert task['config']['max_steps']==CAPS[task['suite']]
    assert read(Path(task['summary']))['run_config']==task['config']
    frozen = read(task_root/'source_files.json')
    source = Path(manifest['source_snapshot'])/'libero_system'
    assert {str(p.relative_to(source)): sha(p.read_bytes()) for p in source.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}==frozen
    raw = (task_root/'episodes.jsonl').read_bytes()
    assert sha(raw)==validation['episodes_jsonl_sha256']
    originals = {json.loads(line)['episode_id']: (json.loads(line), line) for line in raw.splitlines()}
    assert len(raw.splitlines())==len(originals)==10
    assert set(originals)==set(task['episodes'])
    new_rows = []
    for row, line in originals.values():
        assert (row['key']['suite'], row['key']['task_id']) == (task['suite'], task['task_id'])
        assert row['seed']==7 and 0<=row['steps']<=CAPS[task['suite']]
        assert row['policy_status']!='exception' and type(row['evaluator_success']) is bool
        assert row['route_trace']['evaluator_isolation']['scoring']=='external_sticky_any_success'
        assert all(Path(path).is_file() for path in row['video_paths'].values())
        new_rows.append(project(row, line, manifest['run_name'], manifest['source_tree_sha256']))
    assert {row['init_id'] for row in new_rows}==set(range(10))
    assert sum(row['success'] for row in new_rows)==status['overall']['successes']
    detail = {'suite': task['suite'], 'task_id': task['task_id'], 'episodes': 10,
              'successes': sum(row['success'] for row in new_rows),
              'batch_manifest': manifest, 'batch_validation': validation,
              'batch_manifest_sha256': sha((task_root/'manifest.json').read_bytes()),
              'batch_elapsed_wall_s': status['elapsed_wall_s'],
              'baseline_reference': 'full400_reference/provenance.json'}
    return new_rows, originals, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-batch', required=True, type=Path)
    parser.add_argument('--task-batch', required=True, action='append', type=Path,
                        help='Repeat for every retained targeted retest, including earlier task updates.')
    parser.add_argument('--output-dir', type=Path, default=PAPER/'data')
    args = parser.parse_args()
    new_rows, originals, replacements = [], {}, []
    for task_root in args.task_batch:
        rows, lines, detail = validate_task(task_root.resolve())
        assert not (originals.keys() & lines.keys()), 'A task may be replaced only once.'
        new_rows.extend(rows)
        originals.update(lines)
        replacements.append(detail)
    refreshed_count = len(new_rows)
    retained_count = 400 - refreshed_count

    with tempfile.TemporaryDirectory(prefix='anchor-reference-') as temp:
        reference = Path(temp)
        subprocess.run(['python3', str(PAPER/'scripts/capture_results.py'),
                        '--batch-dir', str(args.baseline_batch.resolve()),
                        '--output-dir', str(reference)], check=True)
        base_meta = read(reference/'provenance.json')
        base_rows = [json.loads(line) for line in (reference/'episodes.jsonl').read_text().splitlines()]
        base_raw = gzip.decompress((reference/'raw_episodes.jsonl.gz').read_bytes())
        base_originals = {json.loads(line)['episode_id']: line for line in base_raw.splitlines()}
        unchanged = [row for row in base_rows if row['episode_id'] not in originals]
        assert len(unchanged)==retained_count
        assert all(row['instruction']==next(old['instruction'] for old in base_rows
                   if old['episode_id']==row['episode_id']) for row in new_rows)
        for row in unchanged:
            row.update(reused=True, source_campaign=base_meta['run_name'],
                       controller_source_sha256=base_meta['controller_source_sha256'])
        combined = sorted(unchanged+new_rows, key=lambda row:(row['suite'],row['task_id'],row['init_id']))
        assert len({row['episode_id'] for row in combined})==400
        assert set(Counter((row['suite'],row['task_id']) for row in combined).values())=={10}
        combined_raw = b''.join((originals[row['episode_id']][1] if row['episode_id'] in originals
                               else base_originals[row['episode_id']])+b'\n' for row in combined)
        archive = gzip.compress(combined_raw, mtime=0)
        payload = ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in combined).encode()
        out = args.output_dir.resolve()
        out.mkdir(parents=True,exist_ok=True)
        saved = out/'full400_reference'
        saved.mkdir(exist_ok=True)
        for name in ('episodes.jsonl','raw_episodes.jsonl.gz','provenance.json'):
            if (saved/name).exists():
                assert (saved/name).read_bytes()==(reference/name).read_bytes(), 'Historical reference must remain immutable.'
            else:
                shutil.copy2(reference/name,saved/name)
        (out/'episodes.jsonl').write_bytes(payload)
        (out/'raw_episodes.jsonl.gz').write_bytes(archive)
        for detail in replacements:
            detail['baseline_successes'] = sum(row['success'] for row in base_rows
                if (row['suite'], row['task_id']) == (detail['suite'], detail['task_id']))
        source_counts = Counter(row['controller_source_sha256'] for row in combined)
        source_campaigns = {}
        for row in combined:
            source_campaigns.setdefault(row['controller_source_sha256'], set()).add(row['source_campaign'])
        version_count = len(source_counts)
        latest_validation = max(item['batch_validation']['validated_at'] for item in replacements)
        provenance = {
            'run_name': 'full400_reference_plus_targeted_retests', 'state':'completed', 'complete':True,
            'evaluation_kind':'task_replacement', 'status_updated_at': latest_validation,
            'recorded_episodes':400, 'expected_episodes':400,
            'protocol':{**base_meta['protocol'],'new_episodes':refreshed_count,'reused_episodes':retained_count},
            'controller_source_sha256':None,
            'controller_sources':[
                {'run_names': sorted(source_campaigns[source]), 'source_tree_sha256': source,
                 'included_episodes': count} for source, count in source_counts.items()],
            'baseline_total_successes':sum(row['success'] for row in base_rows),
            'baseline_wall_elapsed_s':base_meta['wall_elapsed_s'],
            'task_replacements':replacements,
            'episodes_projection_sha256':sha(payload),
            'raw_episode_archive':{'file':'raw_episodes.jsonl.gz','sha256':sha(archive),
                'uncompressed_sha256':sha(combined_raw),'records':400},
            'projection_note':f'Combined task records: {retained_count} historical episodes from the original frozen full campaign and {refreshed_count} targeted retest episodes. The records contain {version_count} controller versions. This is not a fresh full-400 evaluation of the updated controller. Each record identifies its source campaign, controller checksum and original JSONL-line hash.',
            'validation':{'coverage_passed':True,'issues':[],'episodes':400,
                'episodes_jsonl_sha256':sha(combined_raw),'validated_at':latest_validation,
                'scope':f'Combined coverage and source-record integrity; {version_count} controller versions.'},
            'wall_elapsed_s':sum(item['batch_elapsed_wall_s'] for item in replacements),
            'wall_elapsed_scope':'Sum of the targeted retest dispatcher durations; historical campaign excluded.',
        }
        write(out/'provenance.json',provenance)
        print(json.dumps({'targeted_results':[
            {key: item[key] for key in ('suite','task_id','successes','episodes')} for item in replacements],
            'combined_successes':sum(row['success'] for row in combined),
            'fresh_episodes':refreshed_count,'historical_episodes':retained_count},indent=2))



if __name__=='__main__':
    main()
