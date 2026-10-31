"""Local persistent nonce ledger for the software trust-boundary prototype."""
import sqlite3, os, json, hmac, hashlib
from pathlib import Path
class FreshnessError(ValueError):pass
class KeyService:
 def __init__(self,path):
  self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
  with self.connect() as c:
   c.execute('CREATE TABLE IF NOT EXISTS nonces (scope BLOB PRIMARY KEY)')
   c.execute('CREATE TABLE IF NOT EXISTS artifacts (scope BLOB PRIMARY KEY, content BLOB NOT NULL)')
  os.chmod(self.path,0o600)
 def connect(self):return sqlite3.connect(self.path,timeout=60,isolation_level='IMMEDIATE')
 def claim(self,scope):
  try:
   with self.connect() as c:c.execute('INSERT INTO nonces VALUES (?)',(scope,))
  except sqlite3.IntegrityError as e:raise FreshnessError('private request nonce already used') from e
 def public(self,scope,payload):
  content=hashlib.sha3_256(payload).digest()
  with self.connect() as c:
   c.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?)',(scope,content))
   prior=c.execute('SELECT content FROM artifacts WHERE scope=?',(scope,)).fetchone()[0]
   if not hmac.compare_digest(prior,content):raise FreshnessError('public artifact content changed for the same identity')
def scope(*parts):
 # Length-prefix every component; variable principal/nonce concatenation is ambiguous.
 out=b''
 for p in parts:
  p=p if isinstance(p,bytes) else str(p).encode()
  out+=len(p).to_bytes(8,'big')+p
 return out
