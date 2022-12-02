import ed25519
import hashlib
import base64
 
# Example of a typical "response" object
response = {
  'status': 200,
  'body': """{"data":[{"id":"63ac9241-0bff-4a64-83bb-df6aec781b0e","type":"licenses","attributes":{"name":"Ed25519 License","key":"key/eyJhY2NvdW50Ijp7ImlkIjoiYmY5YjUyM2YtZGQ2NS00OGEyLTk1MTItZmI2NmJhNmMzNzE0In0sInByb2R1Y3QiOnsiaWQiOiI5NTYxYzdkMC1mYzczLTRjOTQtYTZlZC0xY2M3MmEzZTAzNzYifSwicG9saWN5Ijp7ImlkIjoiNTQ2ZTc0OGUtZjhmYS00ODBjLWJjMDItNjYzMjdjOGZkMGZmIiwiZHVyYXRpb24iOm51bGx9LCJ1c2VyIjpudWxsLCJsaWNlbnNlIjp7ImlkIjoiNjNhYzkyNDEtMGJmZi00YTY0LTgzYmItZGY2YWVjNzgxYjBlIiwiY3JlYXRlZCI6IjIwMjEtMDYtMDFUMTU6MTM6NTMuMjUzWiIsImV4cGlyeSI6bnVsbH19.4ctbpwScfuuxkcynfPbmDrfwJojEHBc7ixgdSy9OKZtIRWEatzbWez3P1UwMhf7fMHXffIdUg5Nb41zqqjRqAA==","expiry":null,"uses":0,"suspended":false,"scheme":"ED25519_SIGN","encrypted":false,"strict":false,"floating":false,"concurrent":false,"protected":false,"maxMachines":1,"maxCores":null,"maxUses":null,"requireHeartbeat": false,"requireCheckIn":false,"lastValidated":"2021-06-04T17:00:58.680Z","lastCheckIn":null,"nextCheckIn":null,"metadata":{},"created":"2021-06-01T15:13:53.253Z","updated":"2021-06-04T17:00:58.680Z"},"relationships":{"account":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714"},"data":{"type":"accounts","id":"bf9b523f-dd65-48a2-9512-fb66ba6c3714"}},"product":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/product"},"data":{"type":"products","id":"9561c7d0-fc73-4c94-a6ed-1cc72a3e0376"}},"policy":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/policy"},"data":{"type":"policies","id":"546e748e-f8fa-480c-bc02-66327c8fd0ff"}},"user":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/user"},"data":null},"machines":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/machines"},"meta":{"cores":0,"count":0}},"tokens":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/tokens"}},"entitlements":{"links":{"related":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e/entitlements"}}},"links":{"self":"/v1/accounts/bf9b523f-dd65-48a2-9512-fb66ba6c3714/licenses/63ac9241-0bff-4a64-83bb-df6aec781b0e"}}]}""",
  'headers': {
    'keygen-signature': 'keyid="bf9b523f-dd65-48a2-9512-fb66ba6c3714", algorithm="ed25519", signature="KhgcM+Ywv+DnQj4gE+DqWfNTM2TG5wfRuFQZ/zW48ValZuCHEu1h95Uyldqe7I85sS/QliCiRAF5QfW8ZN2vAw==", headers="(request-target) host date digest"',
    'digest': 'sha-256=827Op2un8OT9KJuN1siRs5h6mxjrUh4LJag66dQjnIM=',
    'date': 'Wed, 09 Jun 2021 16:08:15 GMT',
  }
}
 
# In a real scenario, we would parse the signature param from
# the `keygen-signature` header. But for brevity...
response_sig  = 'KhgcM+Ywv+DnQj4gE+DqWfNTM2TG5wfRuFQZ/zW48ValZuCHEu1h95Uyldqe7I85sS/QliCiRAF5QfW8ZN2vAw=='
response_body = response['body'].encode()
 
# Sign the response body using SHA-256
digest_bytes = hashlib.sha256(response_body).digest()
enc_digest   = base64.b64encode(digest_bytes).decode()
 
# Reconstruct the signing data
signing_data = \
  '(request-target): get /v1/accounts/keygen/licenses?limit=1\n' \
  'host: api.keygen.sh\n' \
  'date: ' + response['headers']['date'] + '\n' \
  'digest: sha-256=' + enc_digest
 
print(signing_data)

# Verify the response signature
hex_verify_key = '799efc7752286e6c3815b13358d98fc0f0b566764458adcb48f1be2c10a55906'
verify_key     = ed25519.VerifyingKey(hex_verify_key.encode(), encoding='hex')
try:
  verify_key.verify(response_sig, signing_data.encode(), encoding='base64')
 
  print('signature is good')
except ed25519.BadSignatureError:
  print('signature is bad')