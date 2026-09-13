#!/usr/bin/env python3
"""Record completed development outcomes without constructing a mixed score."""
from datetime import datetime, timezone
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
runs = []
for batch in sorted((repo / 'runtime/route_b_90').iterdir()):
    if not (batch / 'manifest.json').exists() or not (batch / 'status.json').exists():
        continue
    manifest = json.loads((batch / 'manifest.json').read_text())
    status = json.loads((batch / 'status.json').read_text())
    outcomes = []
    for path in sorted((batch / 'shards').glob('*/episodes.jsonl')):
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            outcomes.append({'episode_id': row['episode_id'], 'success': row['evaluator_success'],
                             'steps': row['steps'], 'failure': row['failure']})
    runs.append({'label': batch.name, 'state': status['state'],
                 'source_tree_sha256': manifest['source_tree_sha256'],
                 'protocol': manifest['protocol'], 'outcomes': outcomes,
                 'validated': (json.loads((batch / 'final-validation.json').read_text())
                               if (batch / 'final-validation.json').exists() else None),
                 'stop_reason': status.get('stop_reason')})
record = {'updated_at': datetime.now(timezone.utc).isoformat(),
          'baseline': {'successes': 347, 'episodes': 400},
          'acceptance': 'one frozen source, 400 fresh episodes, at least 360 external successes',
          'full_400_accepted': False, 'development_runs': runs}
(Path(__file__).parent / 'development_record.json').write_text(json.dumps(record, indent=2)+'\n')
print(f'Recorded {len(runs)} development batches; no mixed candidate score is reported.')
