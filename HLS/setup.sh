#!/bin/bash

if [ -f "/etc/os-release" ]; then
    . /etc/os-release

    if [ "$ID" == "ubuntu" ]; then
        # Add Docker's official GPG key:
        sudo apt-get update
        sudo apt-get install ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg

        # Add the repository to Apt sources:
        echo \
          "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
          sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update

        # Install Docker
        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

        # Pull Merlin Docker
        sudo docker pull ghcr.io/ucla-vast/merlin-ucla:latest
    elif [ "$ID" == "centos" ]; then
        # Setup the repository
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

        # Install Docker Engine
        sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

        # Start Docker
        sudo systemctl start docker

        # Pull Merlin Docker
        sudo docker pull ghcr.io/ucla-vast/merlin-ucla:latest
    else
        echo "Unsupported distribution: $ID"
        exit 1
    fi
else
    echo "Unable to determine the distribution."
    exit 1
fi