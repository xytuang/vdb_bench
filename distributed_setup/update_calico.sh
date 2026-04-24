#!/bin/bash

kubectl set env daemonset/calico-node -n kube-system IP_AUTODETECTION_METHOD=interface=enp23s0f0np0
kubectl rollout status daemonset/calico-node -n kube-system
