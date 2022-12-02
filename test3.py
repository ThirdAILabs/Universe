import ed25519
import hashlib
import base64
 
# Reconstruct the signing data
signing_data = """(request-target): post /v1/accounts/thirdai/licenses/actions/validate-key
host: api.keygen.sh
date: Fri, 02 Dec 2022 18:18:52 GMT
digest: sha-256=1XDmdNDvjyEREwmY3sqMFjXpTi78VIP3mwlxKpzPGGM="""
 
# Verify the response signature
response_sig = "mq6VVJpyCCHE1TXIWrFyvQAPdfGRPMUgu8H0Km2wA9gEzHQCcmSTtkrHK1TyuQpckxv5QJP0Z9IBPBnq6cirAw=="
print(len(signing_data), len(response_sig))
hex_verify_key = '9adbfd881d363d31c156c348996892dd0183a7e4540dc001cb7c2eec5a7966ae'
verify_key = ed25519.VerifyingKey(hex_verify_key.encode(), encoding='hex')
print(help(verify_key))
try:
  verify_key.verify(response_sig, signing_data.encode(), encoding='base64')
 
  print('signature is good')
except ed25519.BadSignatureError:
  print('signature is bad')