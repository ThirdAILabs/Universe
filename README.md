# ThirdAI Platform Driver Script

This repository contains a script, `driver.sh`, which automates the deployment of the ThirdAI platform using Ansible. This document provides instructions on how to use the script and what it does.

## Prerequisites

Before running the script, ensure the following are installed on your machine:

1. **Ansible**: The script automatically installs Ansible if it's not already installed.
2. **Bash**: Ensure you have a Bash shell environment available to run the script.
3. **Configuration File**: A `config.yml` file is required to provide necessary configurations for the Ansible playbooks.

## Files in the Package

- `driver.sh`: The main script to automate the deployment.
- `config.yml`: The configuration file required for Ansible to deploy the platform.
- `models/Llama-3.2-1B-Instruct-f16.gguf`: The generation model used by the platform.
- `platform`: This contains Ansible playbooks and other necessary files.

## How to Run `driver.sh`

### Step-by-Step Instructions

1. **Download and Extract the Package**:
   
   After downloading the `thirdai-platform-package.tar.gz`, extract it:

   ```bash
   mkdir -p my_folder
   tar -xzvf thirdai-platform-package.tar.gz -C my_folder
   ```

   This will extract the following files and directories:
   - `driver.sh`
   - `config.yml`
   - `platform/`
   - `models/Llama-3.2-1B-Instruct-f16.gguf`


2. **Run the Script with Various Options**:
   
   The `driver.sh` script supports several options to customize its behavior. Below are the available options and examples of how to use them:

   - **Basic Usage**:
     
     Run the script with the default configuration:

     ```bash
     ./driver.sh ./config.yml
     ```

   - **Enable Verbose Mode**:
     
     Use the `-v` or `--verbose` flag to enable verbose output, which provides more detailed logs during execution.

     ```bash
     ./driver.sh --verbose ./config.yml
     ```

     Or using the short flag:

     ```bash
     ./driver.sh -v ./config.yml
     ```

   - **Run Cleanup Operations**:
     
     Use the `--cleanup` flag to run the cleanup playbook, which removes existing deployments or resources as defined in the playbook.

     ```bash
     ./driver.sh --cleanup ./config.yml
     ```

   - **Onboard New Clients**:
     
     Use the `--onboard_clients` flag to run the onboarding playbook, which sets up new client configurations. You will be prompted to provide the path to the new client configuration file.

     ```bash
     ./driver.sh --onboard_clients ./config.yml
     ```

   - **Combine Multiple Options**:
     
     You can combine multiple options to perform more complex operations. For example, to run the onboarding playbook in verbose mode:

     ```bash
     ./driver.sh --onboard_clients --verbose ./config.yml
     ```

     Or to enable cleanup with verbose output:

     ```bash
     ./driver.sh --cleanup --verbose ./config.yml
     ```

3. **Understanding the Script Flags and Behavior**
   
   - **Verbose Mode (`-v` or `--verbose`)**:
     
     When enabled, the script provides detailed logs (`-vvvv` level) during the execution of Ansible playbooks. This is useful for debugging and understanding the steps being performed.

   - **Cleanup Mode (`--cleanup`)**:
     
     Triggers the execution of the `test_cleanup.yml` Ansible playbook, which is designed to remove existing deployments or resources. This is useful for resetting the environment before a fresh deployment.

   - **Onboard Clients (`--onboard_clients`)**:
     
     Initiates the onboarding process for new clients by running the `onboard_clients.yml` Ansible playbook. The script will prompt you to provide the path to the new client configuration file (`new_client_config.yml`).

4. **Examples of Common Use Cases**

   - **Deploy with Default Settings**:

     ```bash
     ./driver.sh ./config.yml
     ```

   - **Deploy with Verbose Output**:

     ```bash
     ./driver.sh --verbose ./config.yml
     ```

   - **Clean Up Existing Deployments**:

     ```bash
     ./driver.sh --cleanup ./config.yml
     ```

   - **Onboard New Clients with Verbose Output**:

     ```bash
     ./driver.sh --onboard_clients --verbose ./config.yml
     ```

   - **Full Operation: Onboard Clients and Enable Verbose**:

     ```bash
     ./driver.sh --onboard_clients --verbose ./config.yml
     ```

5. **Additional Notes**

   - **Configuration File**:
     
     Ensure that the `config.yml` file is correctly configured with all necessary settings for your deployment. You can use the default `config.yml` provided or supply your own.

   - **Model Files**:
     
     The script checks for the presence of model files in the `pretrained-models/` directory. If not found, it will proceed without them but will issue a warning.

   - **Docker Images**:
     
     The script searches for a `docker_images-*` directory. If not found, it will proceed without Docker images and issue a warning.

   - **Ansible Dependencies**:
     
     The `driver.sh` script ensures that Ansible and the required Ansible Galaxy collections are installed before proceeding. If Ansible is not found, it attempts to install it based on the operating system.

By following these instructions and utilizing the available options, you can effectively manage deployments, perform cleanups, and onboard new clients using the `driver.sh` script.


### Instruction to migrating to a different public IP/DNS

When changing the public IP of your Cluster, follow these steps to update the settings and ensure proper functionality:

---

4. **Steps to Update Frontend URL**

   - To access the admin console, follow these steps:

      1. **Set up Port Forwarding**  
         Open a terminal on your local machine and run the following command:  
         ```bash
         sudo ssh -i <public-key> -L 443:<PRIVATE_IP_OF_MACHINE>:443 <USERNAME>@<NEW_PUBLIC_IP>
         ```  

      2. **Access the Admin Console**  
         Once port forwarding is successfully set up, open your browser on the local machine and navigate to:  
         ```  
         https://localhost/keycloak/admin/master/console/  
         ```  

         You should now be redirected to the admin console.

   - In the **Keycloak Admin Console**, go to:
      - Select the realm: `Thirdai-Platform`.
      - Navigate to **Realm Settings → General**.

   - Update the **Frontend URL** to:
     ```
     https://{newPublicIP}/keycloak
     ```
   - If you dont have the access to older admin console, then you may need to do change the env var `KC_HOSTNAME` and `KC_HOSTNAME_ADMIN` to new public IP in the Keycloak Job, restart it before seeing the change. 


5. **What Happens During Execution**:
   
   - The script checks for the installation of Ansible. If Ansible is not installed, the script will install it automatically.
   - The script verifies if the model folder (`gen-ai-models/`) is present. If the folder is not found, the script issues a warning but proceeds with the playbook execution.
   - The script searches for a `docker_images` folder and warns if it's not found, but proceeds with the playbook execution.
   - The script then navigates to the `platform/` directory and runs the `test_deploy.yml` Ansible playbook using the provided `config.yml`, the model path, the Docker images path as extra variables.

### Troubleshooting

- **Permission Denied**: If you encounter a "permission denied" error while running the script, ensure that the script has executable permissions by running the `chmod +x driver.sh` command. If you are using a `.pem` key for SSH, make sure the key file's permission is set to `400` by running `chmod 400 your-key.pem`.
- **Config File Not Found**: Ensure that the path to the `config.yml` file is correct and that the file exists at the specified location.
- **Ansible Errors**: If Ansible encounters errors during execution, review the output carefully. Ensure that your system has internet access for package installation and model downloading.