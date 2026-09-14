#!/usr/bin/env python3
"""Replace one complete task in a historical result with a validated fresh retest.

The output is explicitly a combined record set from two controller versions,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-batch', required=True, type=Path)
    parser.add_argument('--task-batch', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=PAPER/'data')
    args = parser.parse_args()
    task_root = args.task_batch.resolve()
    manifest = read(task_root/'manifest.json')
    status = read(task_root/'status.json')
    assert status['state']=='completed', 'The targeted retest is not complete.'
    validation = read(task_root/'final-validation.json')
    assert validation['coverage_passed'] and not validation['issues']
    assert len(manifest['tasks'])==1
    task = manifest['tasks'][0]
    assert (task['suite'], task['task_id'])==('libero_10', 9)
    protocol = manifest['protocol']
    assert protocol['official_init_ids']==list(range(10))
    assert protocol['seed']==7 and protocol['episodes_per_task']==10
    assert protocol['new_episodes']==10 and protocol['reused_episodes']==0
    assert protocol['scoring']=='external_sticky_any_success'
    assert task['config']['max_steps']==520
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
        assert row['key']['suite']=='libero_10' and row['key']['task_id']==9
        assert row['seed']==7 and 0<=row['steps']<=520
        assert row['policy_status']!='exception'
        assert row['route_trace']['evaluator_isolation']['scoring']=='external_sticky_any_success'
        assert all(Path(path).is_file() for path in row['video_paths'].values())
        new_rows.append(project(row, line, manifest['run_name'], manifest['source_tree_sha256']))
    assert {row['init_id'] for row in new_rows}==set(range(10))
    assert sum(row['success'] for row in new_rows)==status['overall']['successes']

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
        assert len(unchanged)==390
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
        provenance = {
            'run_name': 'full400_reference_plus_long09_retest', 'state':'completed', 'complete':True,
            'evaluation_kind':'task_replacement', 'status_updated_at': status['updated_at'],
            'recorded_episodes':400, 'expected_episodes':400,
            'protocol':{**base_meta['protocol'],'new_episodes':10,'reused_episodes':390},
            'controller_source_sha256':None,
            'controller_sources':[
                {'run_name':base_meta['run_name'],'source_tree_sha256':base_meta['controller_source_sha256'],'included_episodes':390},
                {'run_name':manifest['run_name'],'source_tree_sha256':manifest['source_tree_sha256'],'included_episodes':10}],
            'task_replacement':{'suite':'libero_10','task_id':9,'episodes':10,
                'successes':sum(row['success'] for row in new_rows),
                'baseline_total_successes':sum(row['success'] for row in base_rows),
                'baseline_wall_elapsed_s':base_meta['wall_elapsed_s'],
                'baseline_successes':sum(row['success'] for row in base_rows if row['episode_id'] in originals),
                'batch_manifest':manifest, 'batch_validation':validation,
                'batch_manifest_sha256':sha((task_root/'manifest.json').read_bytes()),
                'batch_elapsed_wall_s':status['elapsed_wall_s'],
                'baseline_reference':'full400_reference/provenance.json'},
            'episodes_projection_sha256':sha(payload),
            'raw_episode_archive':{'file':'raw_episodes.jsonl.gz','sha256':sha(archive),
                'uncompressed_sha256':sha(combined_raw),'records':400},
            'projection_note':'Combined task records: 390 historical episodes from the original frozen full campaign and 10 fresh Long 09 episodes from one updated controller. This is not a fresh full-400 evaluation of the updated controller. Each record identifies its source campaign, controller checksum and original JSONL-line hash.',
            'validation':{'coverage_passed':True,'issues':[],'episodes':400,
                'episodes_jsonl_sha256':sha(combined_raw),'validated_at':validation['validated_at'],
                'scope':'Combined coverage and source-record integrity; two controller versions.'},
            'wall_elapsed_s':status['elapsed_wall_s'],
        }
        write(out/'provenance.json',provenance)
        print(json.dumps({'long09_successes':sum(row['success'] for row in new_rows),
                          'combined_successes':sum(row['success'] for row in combined),
                          'fresh_episodes':10,'historical_episodes':390},indent=2))


if __name__=='__main__':
    main()
