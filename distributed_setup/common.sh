#!/bin/bash

# Set hostnames. Make sure /etc/hosts is already arranged in increasing order of node numbers
# cp /etc/hosts /etc/hosts.old
echo "Setting hostnames"
# names=("k8s-vdbbench" "k8s-master" "k8s-querynode1" "k8s-querynode2" "k8s-querynode3" "k8s-minio" "k8s-etcd" "k8s-misc")
names=("k8s-vdbbench" "k8s-master" "k8s-querynode" "k8s-minio" "k8s-etcd" "k8s-misc")

# Use a counter to skip the first line
i=0
while IFS= read -r line; do
    if [ $i -eq 0 ]; then
        echo "$line"
    else
        # Append the first word from the array
        echo "$line ${names[i - 1]}"
    fi
    ((i++))
done < /etc/hosts | sudo tee /etc/hosts.new
sudo mv /etc/hosts.new /etc/hosts

echo "Disabling swap"
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

echo "Enabling kernel modules and configure sysctl"
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay # container storage
sudo modprobe br_netfilter # network bridging

echo "Allow Linux iptables to see bridged traffic and enable IP forwarding"
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system

echo "Installing containerd runtime" # containerd is the container runtime
sudo apt-get update
sudo apt-get install -y containerd
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml # Create default configuration
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml # Modify default config to use SystemCgroup driver
sudo systemctl restart containerd
sudo systemctl enable containerd

echo "Installing K8s packages"
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl # apt-mark hold prevents upgrades that could break the cluster

