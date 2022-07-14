import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'ray[default]'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pyyaml'])


# process output with an API in the subprocess module:
reqs = subprocess.check_output([sys.executable, '-m', 'pip',
'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print('Installed Packages:')
print(installed_packages)