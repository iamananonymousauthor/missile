echo "========== Update NVIDIA Driver Patch =========="
kernel_version=$(uname -r)
nvidia_driver_version=535.129.03
echo "Linux kernel_version: $kernel_version"
echo "NVIDIA driver version: $nvidia_driver_version"
echo "================================================"

echo "************ Preparing for updating ************"
# Disable graphic desktop
sudo init 3
# Unload all NVIDIA's dkms modules
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
echo "******** Prepare for updating: Finished ********"

echo "%%%%%%%%%%%%% Updating DKMS modules %%%%%%%%%%%%"
sudo dkms remove nvidia/$nvidia_driver_version --all
sudo dkms install nvidia/$nvidia_driver_version
sudo cp *.ko /usr/lib/modules/$kernel_version/updates/dkms/
sudo modprobe nvidia
echo "%%%%%%%%% Update DKMS modules: Finished %%%%%%%%"