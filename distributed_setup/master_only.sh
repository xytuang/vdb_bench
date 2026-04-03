#!/bin/bash

echo "Initializing cluster with kubeadm"
# Set internal IP range for pods with --pod-network-cidr 
# 10.244.0.0/16 because Calico CNI uses this for default
# Provide stable endpoint for workers to find API server with --control-plane-endpoint
# k8s-master (intuitive)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --control-plane-endpoint=k8s-master



echo "Configuring kubectl access"
# Copy the admin config file into the user's home directory.
# kubectl needs this file to get credentials and address needed to communicate with cluster
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

echo "Setting up pod network"
# Get the Calico manifest that contains all the necessary components. K8s starts these components
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml




