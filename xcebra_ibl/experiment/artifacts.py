"""Atomic artifacts, integrity-checked completion and portable fingerprints."""
import hashlib
import json
from pathlib import Path
import signal
from contextlib import contextmanager


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete(directory, payload, marker='complete.json'):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != marker and not p.name.endswith('.tmp')
             and 'checkpoints' not in p.relative_to(directory).parts}
    write_json(directory / marker, dict(payload, artifacts=files))


def verified(directory, marker='complete.json'):
    path = Path(directory) / marker
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not payload.get('artifacts'):
        raise ValueError(f'Completion marker lacks integrity manifest: {path}')
    for name, expected in payload['artifacts'].items():
        artifact = Path(directory) / name
        if not artifact.resolve().is_relative_to(Path(directory).resolve()):
            raise ValueError("Artifact path escapes its output directory")
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f'Missing or corrupted artifact: {artifact}. Use a fresh output directory or restore the file.')
    return payload


@contextmanager
def interruption_status(output):
    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}; completed variables remain reusable')
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    except (InterruptedError, KeyboardInterrupt) as exc:
        write_json(Path(output) / 'interrupted.json', {'reason': str(exc)})
        raise
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


@contextmanager
def warning_report(directory):
    """Preserve warning counts in an artifact instead of flooding the training log."""
    import warnings
    from collections import Counter
    with warnings.catch_warnings(record=True) as captured:
        try:
            yield
        finally:
            counts = Counter((w.category.__name__, str(w.message), w.filename, w.lineno) for w in captured)
            write_json(Path(directory)/'warnings.json', {
                'emitted_warning_count':len(captured),
                'warnings':[dict(category=k[0],message=k[1],filename=k[2],line=k[3],count=n) for k,n in counts.items()],
                'note':'Warnings captured without changing filters. Finite-value checks remain active.'})
            if captured:
                print(f'{len(captured)} numerical/library warnings recorded in {Path(directory)/"warnings.json"}',flush=True)
