package main

import (
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
)

// It's usually not good practice to check in private keys, but there isn't
// really a better solution here, we are going to be compiling this into a
// binary and having people run the binary.
var privateKeyBuf = []byte(`
-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIFQ8L88G0enbpDW4fZzayY373Wi/jpT5BrZZPQ15xd4A
-----END PRIVATE KEY-----
`)

var publicKeyBuf = []byte(`
-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAqA9j+Pk81yUz7FPZfg94bez6m1j8j1jiLctTjmB2s7w=
-----END PUBLIC KEY-----
`)

var privateKey = GetPrivateKey()

var publicKey = GetPublicKey()

func GetPrivateKey() ed25519.PrivateKey {
	privateKey, _ := pem.Decode(privateKeyBuf)
	if privateKey == nil {
		panic(fmt.Errorf("no pem block found"))
	}

	key, err := x509.ParsePKCS8PrivateKey(privateKey.Bytes)
	if err != nil {
		panic(err)
	}
	edKey, ok := key.(ed25519.PrivateKey)
	if !ok {
		panic(fmt.Errorf("key is not ed25519 key"))
	}
	return ed25519.PrivateKey(edKey)
}

func GetPublicKey() ed25519.PublicKey {
	publicKey, _ := pem.Decode(publicKeyBuf)
	if publicKey == nil {
		panic(fmt.Errorf("no pem block found"))
	}

	key, err := x509.ParsePKIXPublicKey(publicKey.Bytes)
	if err != nil {
		panic(err)
	}
	edKey, ok := key.(ed25519.PublicKey)
	if !ok {
		panic(fmt.Errorf("key is not ed25519 key"))
	}
	return ed25519.PublicKey(edKey)
}

// Takes in a string, turns it into bytes, signs it, and returns a base64 encoding
// of the signature
func Sign(toSign string) string {
	bytesIn := []byte(toSign)
	bytesOut := ed25519.Sign(privateKey, bytesIn)
	return base64.StdEncoding.EncodeToString(bytesOut)
}
