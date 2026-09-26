"""Real child-process failure checks, no image reconstruction."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
from runner import supervise, unique_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result={}
    sleepy=[sys.executable,'-c','import time; time.sleep(30)']
    for name,cmd,kw in [('timeout',sleepy,{'timeout':.05}),('rss_unavailable',sleepy,{'rss_reader':lambda _:[]}),
                        ('rss_limit',sleepy,{'max_rss':1}),('worker_exception',[sys.executable,'-c','raise RuntimeError("test")'],{}),
                        ('launch_failure',['/nonexistent-baseline-executable'],{})]:
        d=args.output/name
        d.mkdir()
        result[name]=supervise(cmd,d,**kw)
        assert result[name]['returncode']!=0
    assert 'TIME_LIMIT' in result['timeout']['reason']
    assert 'RSS unavailable' in result['rss_unavailable']['reason']
    assert 'RSS_LIMIT' in result['rss_limit']['reason']
    d=args.output/'cancelled'
    d.mkdir()
    timer=threading.Timer(.2,lambda:os.kill(os.getpid(),signal.SIGTERM))
    timer.start()
    result['cancelled']=supervise(sleepy,d)
    timer.join()
    assert 'Cancelled' in result['cancelled']['reason'] and result['cancelled']['returncode']!=0
    try:
        json.loads('{"x":1,"x":2}',object_pairs_hook=unique_json)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate JSON accepted')
    (args.output/'results.json').write_text(json.dumps({'status':'PASS','cases':result},indent=2))
    print('PASS actual timeout/RSS unavailable/RSS limit/worker exception/launch failure/cancellation')


if __name__=='__main__':
    main()
