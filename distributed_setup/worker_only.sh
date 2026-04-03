#!/bin/bash

echo -n "Token from master: "
read token

echo -n "Discovery token ca cert hash: "
read discovery_token_ca_cert_hash
sudo kubeadm join k8s-master:6443 --token $token\
    --discovery-token-ca-cert-hash $discovery_token_ca_cert_hash

