#!/bin/bash

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\033[32m$message\033[0m" ;;
        "red") echo -e "\033[31m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
}

# Function to check if script is run as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_status "red" "Please run this script as root or with sudo"
        exit 1
    fi
}

# Function to clean Docker
clean_docker() {
    print_status "yellow" "Cleaning Docker..."
    
    # Stop all running containers
    if docker ps -q &>/dev/null; then
        print_status "yellow" "Stopping all running containers..."
        docker stop $(docker ps -a -q) 2>/dev/null
    fi

    # Remove all containers
    if docker ps -a -q &>/dev/null; then
        print_status "yellow" "Removing all containers..."
        docker rm $(docker ps -a -q) 2>/dev/null
    fi

    # Remove all images
    if docker images -q &>/dev/null; then
        print_status "yellow" "Removing all Docker images..."
        docker rmi $(docker images -q) -f 2>/dev/null
    fi

    # Remove unused data
    print_status "yellow" "Removing unused Docker data..."
    docker system prune -af --volumes 2>/dev/null

    print_status "green" "Docker cleanup completed"
}

# Function to clean system logs
clean_logs() {
    print_status "yellow" "Cleaning system logs..."

    # Clean journal logs
    if command -v journalctl &>/dev/null; then
        print_status "yellow" "Clearing journalctl logs..."
        journalctl --rotate
        journalctl --vacuum-time=1d
    fi

    # Clean log files in /var/log
    print_status "yellow" "Cleaning /var/log..."
    find /var/log -type f -name "*.log" -exec truncate -s 0 {} \;
    find /var/log -type f -name "*.gz" -delete
    find /var/log -type f -name "*.old" -delete

    print_status "green" "System logs cleaned"
}

# Function to clean package manager cache
clean_package_cache() {
    print_status "yellow" "Cleaning package manager cache..."

    # APT cache (Debian/Ubuntu)
    if command -v apt-get &>/dev/null; then
        print_status "yellow" "Cleaning APT cache..."
        apt-get clean
        apt-get autoremove -y
    fi

    # YUM cache (RHEL/CentOS)
    if command -v yum &>/dev/null; then
        print_status "yellow" "Cleaning YUM cache..."
        yum clean all
    fi

    # DNF cache (Fedora)
    if command -v dnf &>/dev/null; then
        print_status "yellow" "Cleaning DNF cache..."
        dnf clean all
    fi

    print_status "green" "Package manager cache cleaned"
}

# Function to clean user cache
clean_user_cache() {
    print_status "yellow" "Cleaning user cache..."

    # Clean user cache directories
    find /home -type f -name '.DS_Store' -delete
    find /home -type f -name 'Thumbs.db' -delete
    
    # Clean Firefox cache
    find /home -type d -name ".mozilla" -exec rm -rf {}/firefox/**/cache {} \;
    
    # Clean Chrome cache
    find /home -type d -name ".cache/google-chrome" -exec rm -rf {} \;
    
    # Clean general cache
    find /home -type d -name ".cache" -exec rm -rf {}/* \;

    print_status "green" "User cache cleaned"
}

# Function to clean temporary files
clean_temp() {
    print_status "yellow" "Cleaning temporary files..."

    # Clean /tmp directory
    rm -rf /tmp/*
    
    # Clean user temporary directories
    find /home -type d -name "tmp" -exec rm -rf {}/* \;

    print_status "green" "Temporary files cleaned"
}

# Main execution
main() {
    print_status "yellow" "Starting system cleanup..."
    
    # Check if running as root
    check_root

    # Perform cleanup operations
    clean_docker
    clean_logs
    clean_package_cache
    clean_user_cache
    clean_temp

    # Final system sync
    sync

    print_status "green" "System cleanup completed successfully!"
}

# Run main function
main