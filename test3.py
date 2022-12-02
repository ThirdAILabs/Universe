import ed25519
import hashlib
import base64
 
# Reconstruct the signing data
signing_data = \
  '(request-target): post /v1/accounts/thirdai/licenses/actions/validate-key\n' \
  'host: api.keygen.sh\n' \
  'date: Fri, 02 Dec 2022 05:22:10 GMT\n' \
  'digest: sha-256=l3tiDOjJ4lBPfAoS/3KEo8e2MnLWS3Q9iyw3MijS3Pk='
 
print(signing_data)

# Verify the response signature
response_sig = "xSntFHMisR7YCk6vmPVm6Ddsocz4/eJ3d2oe7XhTg3imiVEQlc/fR6qkPM3M4VqTzeLYpBO+EG+rjVBOzj2zCQ=="
hex_verify_key = '9adbfd881d363d31c156c348996892dd0183a7e4540dc001cb7c2eec5a7966ae'
verify_key = ed25519.VerifyingKey(hex_verify_key.encode(), encoding='hex')
try:
  verify_key.verify(response_sig, signing_data.encode(), encoding='base64')
 
  print('signature is good')
except ed25519.BadSignatureError:
  print('signature is bad')