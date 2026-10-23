"""Calibration recovery stress test; not a task-safety evaluation.

This is a new component path. It does not upgrade legacy episodic replay.
Test keys below are public deterministic fixtures, never deployment credentials.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, random, sqlite3, sys, tempfile, time
from datetime import UTC, datetime, timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from aeros.runtime.checkpointed_persona import CheckpointedPersonaStore, TrustedKey, IntegrityError
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_resolver import _apply_event, _risk_cap, effective_persona

def run(seed:int,retention:int, count:int=120)->dict:
 rng=random.Random(seed); core=PersonaCore(risk_appetite='conservative')
 auto,op=[Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f'PUBLIC TEST FIXTURE {seed} {role}'.encode()).digest()) for role in ['auto','operator']]
 keys={'auto':TrustedKey(auto.public_key(),'automatic'),'op':TrustedKey(op.public_key(),'operator')}
 kinds=['skill_confidence_decrease','risk_appetite_calibration_tighten','drift_event','skill_confidence_increase','human_check_threshold_relax','risk_appetite_calibration_relax','human_check_threshold_tighten']
 state=PersonaAdaptive(risk_appetite_actual=min(.5,_risk_cap(core))); latency=[]
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'state.sqlite'; store,anchor=CheckpointedPersonaStore.create(path,'test-agent',core,keys,checkpoint_key_id='op',checkpoint_key=op,retention=retention)
  old_bundle=store.export(anchor); old_anchor=anchor
  for i in range(count):
   kind=kinds[i%7]; magnitude=round(rng.uniform(.005,.035),8)
   delta=({'severity':'low','reason':'synthetic incident'} if kind=='drift_event' else {'magnitude':magnitude,**({'action_kind':'handoff'} if kind.startswith('human_check') else {})})
   ev=PersonaAdaptiveEvent(kind=kind,capability='skill'+str(i%4) if kind.startswith('skill_') else None,delta=delta,timestamp=datetime(2026,10,21,tzinfo=UTC)+timedelta(seconds=i),applied=i%11!=0)
   operator=kind not in ('skill_confidence_decrease','risk_appetite_calibration_tighten','drift_event')
   start=time.perf_counter_ns(); envelope=store.sign_event(ev,anchor,key_id='op' if operator else 'auto',key=op if operator else auto)
   anchor=store.append(envelope,anchor,checkpoint_key_id='op',checkpoint_key=op);latency.append((time.perf_counter_ns()-start)/1e6)
   if ev.applied:state=_apply_event(state,ev,_risk_cap(core))
   assert dataclasses.asdict(store.read(anchor))==dataclasses.asdict(state)
  bundle=store.export(anchor); body=json.loads(bundle)['body']; actual=store.read(anchor)
  with sqlite3.connect(path) as db: rows=db.execute('select count(*) from events').fetchone()[0]
  store.close(); store=CheckpointedPersonaStore.open(path,'test-agent',core,keys,anchor=anchor,retention=retention)
  restarted=store.read(anchor); dest=CheckpointedPersonaStore.restore(Path(tmp)/'dest.sqlite','test-agent',core,keys,bundle,anchor=anchor,retention=retention)
  restored=dest.read(anchor); stale_rejected=False
  try:CheckpointedPersonaStore.restore(Path(tmp)/'stale.sqlite','test-agent',core,keys,old_bundle,anchor=anchor,retention=retention)
  except IntegrityError:stale_rejected=True
  # Matching stale anchor is deliberately accepted: records the rollback limitation.
  stale=CheckpointedPersonaStore.restore(Path(tmp)/'old.sqlite','test-agent',core,keys,old_bundle,anchor=old_anchor,retention=retention)
  rollback_with_anchor_accepted=stale.read(old_anchor).skill_confidence=={}
  state_equal=actual==restarted==restored==state
  assert state_equal and rows<=retention and stale_rejected and rollback_with_anchor_accepted
  for s in (store,dest,stale):s.close()
  return dict(seed=seed,retention=retention,events=count,base_seq=body['base_seq'],retained_rows=rows,
      recovered_equal=state_equal,stale_rejected_current_anchor=stale_rejected,whole_state_and_anchor_rollback_accepted=rollback_with_anchor_accepted,
      export_bytes=len(bundle.encode()),state_sha256=hashlib.sha256(json.dumps(dataclasses.asdict(state),sort_keys=True).encode()).hexdigest(),
      append_including_signing_p50_ms=sorted(latency)[len(latency)//2],append_including_signing_p99_ms=sorted(latency)[-2])

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('choose fresh output; existing evidence is never overwritten')
 rows=[run(seed,retention) for seed in (42,43,44) for retention in (1,8,64)]
 result={'experiment':'e6_checkpoint_recovery','seeds':[42,43,44],'retentions':[1,8,64],'events_per_run':120,'runs':rows,
 'limitations':['Synthetic calibration state only; all checkpoint manifests signed by operator authority in prototype.', 'No integrated legacy CLI/production service or external monotonic anchor deployment.', 'No rate-limit/circuit/EMA evidence state recovery; those mechanisms remain a separate experiment.', 'Timing includes signatures and SQLite commits on this host; not a deployment guarantee.']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'runs':len(rows),'events':sum(r['events'] for r in rows),'all_equal':all(r['recovered_equal'] for r in rows),'max_tail':max(r['retained_rows'] for r in rows)}))
if __name__=='__main__':main()
