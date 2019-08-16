#!/usr/bin/env bash
shopt -s extglob

echo_red() {
    tput setaf 1
    echo "$@"
    tput sgr0
}

echo_green() {
    tput setaf 2
    echo "$@"
    tput sgr0
}

echo_yellow() {
    tput setaf 3
    echo "$@"
    tput sgr0
}

echo_blue() {
    tput setaf 4
    echo "$@"
    tput sgr0
}

ret_check() {
	if [ $? -ne 0 ]; then
		tput setaf 1
		echo "$@"
		tput sgr0
		exit -1;
	fi
}

####################################################################################
#                                  simics keys                                    #
###################################################################################
declare -a simics_keys=(
    ["1000"]="70cdd97a5ed8f877e3f79a2b78fd6c8b"
    ["1001"]="547274b3080c844fb8f1161eedc890bb"
    ["7031"]="410f41f94209a571bdae19c93e07de29"
    ["7047"]="24f6742c111d027b389b821c82c05588"
    ["7127"]="1e03dbf890dc14509f3f1f6ba51d495731b3f081e1942a730420a1b51a6d449a"
)

###################################################################################
# function build_deps_check                                                       #
#                                                                                 #
# parameters: none                                                                #
# returns: nothing, exits on failure                                              #
###################################################################################
build_deps_check() {
    local res=1
    deps=( awk grep python sed )

    echo_blue "Checking dependencies for build"

    for var in "${deps[@]}"
    do
        echo -n "Checking for ${var}..."
        if [[ `which ${var}` == "" ]]; then
            echo_red " missing"
            res=0
        else
            echo_green " found"
        fi
    done

    if [[ res -eq 1 ]]; then
        echo_green "Looks like all prerequisites are there..."
    else
        ret_check "Error: build required components are missing!"
    fi
}

###################################################################################
# function get_external_dependency                                                #
#                                                                                 #
# parameters:                                                                     #
# $1: config_file.  i.e. simicsDIR/config.ini                                     #
#                                                                                 #
# returns: nothing, exits on failure                                              #
###################################################################################

artifact_exit_code() {
	retVal=$?
	p_id=$PPID
	echo " <<<< exit code:  $retVal    >>>>"
        echo " <<<< PPID     :  $p_id      >>>>"

        if [ $retVal -ne 0 ]; then
           echo "<<< [ERROR] Fails at during download artifact  >>>"
           exit $retVal
        fi
}

get_external_dependency() {
	local config_file=$1

	echo "Downloading external dependency according to ${config_file}"
	# cd ${SIMICS_PKG_TMP}
        cd ${SIMICS_PKGS}
	echo "$(date +%H:%M:%S): start time"

	pkg1000=$(awk -F "=" '/pkg1000/ {print $2}' $config_file)
	pkg1001=$(awk -F "=" '/pkg1001/ {print $2}' $config_file)
	pkg7031=$(awk -F "=" '/pkg7031/ {print $2}' $config_file)
	pkg7047=$(awk -F "=" '/pkg7047/ {print $2}' $config_file)
	#sph_bios=$(awk -F "=" '/sph_bios/ {print $2}' $config_file)
	
	#sph_bios=$(echo $sph_bios|tr -d '\n')
	#pkg1000=$(echo $pkg1000|tr -d '\n')
	#pkg1001=$(echo $pkg1001|tr -d '\n')
	#pkg7031=$(echo $pkg7031|tr -d '\n')
	#pkg7047=$(echo $pkg7047|tr -d '\n')


	if [ "${sph_stand_alone,,}" = false ]; then
		pkg7127=$(awk -F "=" '/pkg7127/ {print $2}' $config_file)
		#centos_bios=$(awk -F "=" '/centos_bios/ {print $2}' $config_file)
		#centos_image=$(awk -F "=" '/centos_img/ {print $2}' $config_file)
		
		#pkg7127=$(echo $pkg7127|tr -d '\n')
		#centos_bios=$(echo $centos_bios|tr -d '\n')
		#centos_image=$(echo $centos_image|tr -d '\n')
	fi

	echo ${pkg1000}
	echo ${pkg1001}
	echo ${pkg7031}
	# --------------------
	echo ${pkg7047}
	#echo ${sph_bios}
	# --------------------
	if [ "${sph_stand_alone,,}" = false ]; then
		echo ${pkg7127}
		#echo ${centos_bios}
		#echo ${centos_image}
	fi


	#PKG
	#bash -c "$automation_root/tools/artifact/curl.py download --name simics_5 --path . --props revision='${pkg1000}' revision='${pkg1001}' revision='${pkg7031}' revision='${pkg7047}'  -v"
	artifact_exit_code

	# SPH BIOS
	# bash -c "$automation_root/tools/artifact/curl.py download --name simics_5 --path ${SIMICS_IMAGE} --props revision='${sph_bios}' -v"
	#artifact_exit_code
	#if [ "${sph_stand_alone,,}" = false ]; then
		#bash -c "$automation_root/tools/artifact/curl.py download --name simics_5 --path . --props revision='${pkg7127}' -v"
		#artifact_exit_code
		# BIOS
                #bash -c "$automation_root/tools/artifact/curl.py download --name simics_5 --path ${SIMICS_IMAGE} --props revision='${centos_bios}' revision='${centos_image}' -v"
		#artifact_exit_code
	#fi
        wget -P . ${pkg1000} --no-check-certificate
        echo "pkg1000  successfully"
        wget -P . ${pkg1001} --no-check-certificate
        echo "pkg1001  successfully"
        wget -P . ${pkg7031} --no-check-certificate
        echo "pkg7031  successfully"
        wget -P . ${pkg7047} --no-check-certificate
        echo "pkg7047  successfully"
        wget -P . ${pkg7127} --no-check-certificate
        echo "pkg7127  successfully"

        wait "${pids[@]}"
        ret_check "Failed to download external dependencies from ${config_file}"

	cd -
}

###################################################################################
# function install_pkg                                                            #
#                                                                                 #
# parameters: pkg name (tar file)                                                 #
# returns: nothing, exits on failure                                              #
###################################################################################
install_pkg() {
    local curd=`pwd`
    pkg_file=$1
    if [ -d ${INSTALL_DIR} ]; then
        rm -rf ${INSTALL_DIR}
    fi
    mkdir -p ${INSTALL_DIR}
    cd ${SIMICS_PKGS}

    local pkg_id=`echo ${pkg_file} | grep -o -e "simics-pkg-[0-9]*" | grep -o -e "[0-9]*"`
    local key="${simics_keys[$pkg_id]}"
    echo_blue "Using key $key for package $pkg_id"
    [ "x$key" != "x" ] || ret_check "Key for package $pkd_id is missing!"
    tar -xvpf ${pkg_file}
    cd ${INSTALL_DIR}
    local simics_pkg=`ls -1 . | grep tar.gz`
    ./install-simics.pl -a -p ${SIMICS_DIR} -b old/${simics_pkg} ${key}
    #./install-simics.pl -a -p ${SIMICS_DIR} -b ${simics_pkg}
    cd ${curd}
}


install_pkg_wa() {
    local curd=`pwd`
    pkg_file=$1
    if [ -d ${INSTALL_DIR} ]; then
        rm -rf ${INSTALL_DIR}
    fi
    mkdir -p ${INSTALL_DIR}
    cd ${SIMICS_PKGS}

    local pkg_id=`echo ${pkg_file} | grep -o -e "simics-pkg-[0-9]*" | grep -o -e "[0-9]*"`
    local key="${simics_keys[$pkg_id]}"
    echo_blue "Using key $key for package $pkg_id"
    [ "x$key" != "x" ] || ret_check "Key for package $pkd_id is missing!"
    tar -xvpf ${pkg_file}
    cd ${INSTALL_DIR}
    local simics_pkg=`ls -1 . | grep tar.gz`
    #./install-simics.pl -a -p ${SIMICS_DIR} -b ${simics_pkg} ${key}
    ./install-simics.pl -a -p ${SIMICS_DIR} -b ${simics_pkg}
    cd ${curd}
}


###################################################################################
# function download_head_version                                                  #
#                                                                                 #
# parameters: none                                                                #
# returns: nothing, exits on failure                                              #
###################################################################################
download_head_version() {
    if [ -d ${SIMICS_DIR} ]; then
        rm -fr ${SIMICS_DIR}
    fi
    mkdir -p ${SIMICS_DIR}

    # right now is not download from source
    get_external_dependency $config_file
    cd ${SIMICS_PKG_TMP}

    SIMICS_BASE=`ls -1 . | grep 1000 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Base is: ${SIMICS_BASE}"
    SIMICS_ECLIPSE=`ls -1 . | grep 1001 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Eclipse is: ${SIMICS_ECLIPSE}"
    ECLIPSE_UEFI=`ls -1 . | grep 7031 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Eclipse UEFI is: ${ECLIPSE_UEFI}"
    SIMICS_ICE_LAKE=`ls -1 . | grep 7047 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics ICL is: ${SIMICS_ICE_LAKE}"
    if [ "${sph_stand_alone,,}" = false ]; then
        SIMICS_WHITLEY=`ls -1 . | grep 7127 | tail -1 | tr -d '\r\n'`
       	echo_yellow "Latest Simics Whitley is: ${SIMICS_WHITLEY}"
    fi

    [[ "x${SIMICS_BASE}" != "x" ]] || ret_check "Error: Failed to get Simics base version"
    [[ "x${SIMICS_ECLIPSE}" != "x" ]] || ret_check "Error: Failed to get Simics eclipse version"
    [[ "x${ECLIPSE_UEFI}" != "x" ]] || ret_check "Error: Failed to get Simics uefi version"
    [[ "x${SIMICS_ICE_LAKE}" != "x" ]] || ret_check "Error: Failed to get Simics ice-lake version"

    if [ "${sph_stand_alone,,}" = false ]; then
        [[ "x${SIMICS_WHITLEY}" != "x" ]] || ret_check "Error: Failed to get Simics Whitley version"
    fi

    install_pkg ${SIMICS_PKG_TMP}/${SIMICS_BASE}
    install_pkg ${SIMICS_PKG_TMP}/${SIMICS_ECLIPSE}
    install_pkg ${SIMICS_PKG_TMP}/${ECLIPSE_UEFI}
    install_pkg ${SIMICS_PKG_TMP}/${SIMICS_ICE_LAKE}

    install_pkg_wa ${SIMICS_PKG_TMP}/${SIMICS_WHITLEY}
 
    echo_green "Successfully installed simics packages"
}

###################################################################################
# function Install_Simics                                                  #
#                                                                                 #
# parameters: none                                                                #
# returns: nothing, exits on failure                                              #
###################################################################################
Install_Simics() { 
    cd ${SIMICS_PKGS}
    cp ${BKC_PKGS}/*.tar ./
    SIMICS_BASE=`ls -1 . | grep 1000 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Base is: ${SIMICS_BASE}"
    SIMICS_ECLIPSE=`ls -1 . | grep 1001 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Eclipse is: ${SIMICS_ECLIPSE}"
    ECLIPSE_UEFI=`ls -1 . | grep 7031 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Eclipse UEFI is: ${ECLIPSE_UEFI}"
    SIMICS_ICE_LAKE=`ls -1 . | grep 7047 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics ICL is: ${SIMICS_ICE_LAKE}"
    SIMICS_WHITLEY=`ls -1 . | grep 7127 | tail -1 | tr -d '\r\n'`
    echo_yellow "Latest Simics Whitley is: ${SIMICS_WHITLEY}"

    [[ "x${SIMICS_BASE}" != "x" ]] || ret_check "Error: Failed to get Simics base version"
    [[ "x${SIMICS_ECLIPSE}" != "x" ]] || ret_check "Error: Failed to get Simics eclipse version"
    [[ "x${ECLIPSE_UEFI}" != "x" ]] || ret_check "Error: Failed to get Simics uefi version"
    [[ "x${SIMICS_ICE_LAKE}" != "x" ]] || ret_check "Error: Failed to get Simics ice-lake version"
    [[ "x${SIMICS_WHITLEY}" != "x" ]] || ret_check "Error: Failed to get Simics Whitley version"

    install_pkg ${SIMICS_PKGS}/${SIMICS_BASE}
    install_pkg ${SIMICS_PKGS}/${SIMICS_ECLIPSE}
    install_pkg ${SIMICS_PKGS}/${ECLIPSE_UEFI}
    install_pkg ${SIMICS_PKGS}/${SIMICS_ICE_LAKE}
    install_pkg_wa ${SIMICS_PKGS}/${SIMICS_WHITLEY}
    #if [ "${sph_stand_alone,,}" = false ]; then
    #    install_pkg_wa ${SIMICS_PKG_TMP}/${SIMICS_WHITLEY}
    #fi

    echo_green "Successfully installed simics packages"
}


print_help() {
    echo " "
    echo_green "-----------------------------------------------------------------------------------------------------"
    echo_green "simics root default location: ${simics_root}"
    echo "Usage: $(basename $0) -t <target> -r <simics root> -a <automation root> [see further optional flags]"
    echo " "
    echo "-t Target: install|clean"
    echo "-r Simics installation root path"
    echo "-f select deploy card payload package, sa or e2e. The default setting is e2e"
    #echo "-b Boot option: pci|emmc|skip. Default pci"
    #echo "-m FS mode: jffs2|ramfs|ext3. Default ramfs"
    #echo "-c coral pakage: directory path, Default {simics_root}/../release_artifacts/coral_ubuntu_simics_pakage"
    #echo "-s SPH image: directory path: Default {simics_root}/simics-image"
    #echo "-w Whitley Image directory path Default {simics_root}/simics-image"
    #echo "-g Simics setup config Default {simics_root}/config.ini"
    #echo "-n SPH stand alone. true|false. Default FALSE"
    #echo "-a Automation repo directory path, Default {simics_root}/../aipg_inference_validation-automation"
    echo_green "-----------------------------------------------------------------------------------------------------"
    echo " "
}
#############################################################################################
# main main main main main main main main main main main main main main main main main main #
#############################################################################################
target=""
installer_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ../..
bkc_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ../..

install_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir SPH_BKC
cd ./SPH_BKC
mkdir aipg_inference-simics5
cd ./aipg_inference-simics5
simics_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
sph_image=""
auto_root=""
boot_mode="pci"
card_payload_package="e2e"
fs_mode="ramfs"
sph_stand_alone="FALSE"
coral_pakage=""

whitley_image=""
config_file="FALSE"
num_cores=1


if [ -z $1 ]; then
    echo_red "Missing target. Check \"$(basename $0) -h\" for more details"
    exit -1;
fi

while getopts ":ht:s:a:r:b:c:g:m:w:n:f:" key; do case $key in
  h)
    print_help
    exit 0
    ;;
  f)
    if [ "${OPTARG,,}" == "sa" ]; then
      card_payload_package="sa"
    fi
    ;;

  t)
    target="$OPTARG"
    ;;

  r)
    simics_root=$(readlink -f "$OPTARG")
    ;;

  b)
    if [ "${OPTARG,,}" == "pci" ]; then
      boot_mode="pci"
    elif [ "${OPTARG,,}" == "emmc" ]; then
      boot_mode="emmc"
    elif [ "${OPTARG,,}" == "skip" ]; then
      boot_mode="skip"
    else
      echo_red "Unknown boot mode: $OPTARG"
      exit -1
    fi
    ;;

  m)
    if [ "${OPTARG,,}" == "jffs2" ]; then
        fs_mode="jffs2"
    elif [ "${OPTARG,,}" == "ramfs" ]; then
        fs_mode="ramfs"
    elif [ "${OPTARG,,}" == "ext3" ]; then
        fs_mode="ext3"
    else
      echo_red "Unknown fs mode: $OPTARG"
      exit -1
    fi
    ;;

  c)
    coral_pakage=$(readlink -f "$OPTARG")
    ;;
  w)
    whitley_image=$(readlink -f "$OPTARG")
    ;;
  g)
    config_file=$(readlink -f "$OPTARG")
    ;;
  n)
    if [ "${OPTARG,,}" = "true" ]; then
        sph_stand_alone=true
    else
        sph_stand_alone=false
    fi
    ;;
  s)
    sph_image=$(readlink -f "$OPTARG")
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    print_help
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    print_help
    exit 1
    ;;
esac
done

################################################################
#                             set env                          #
################################################################

build_deps_check

echo " "
echo_yellow "Target: $target"
echo_yellow "Simics root: $simics_root"
# echo_yellow "Automation root: $automation_root"

if [ "${config_file,,}" = "false" ]; then
	config_file="${simics_root}/config.ini"
	echo Download dependency according to: "${config_file}"
else
	echo [override config], Download dependency according to: "${config_file}"
fi
echo " "

export SIMICS_DIR="${simics_root}/simics-5"
export SIMICS_IMAGE="${simics_root}/simics-image"
export SIMICS_PKGS="${simics_root}/simics-5-pkgs"
# export SIMICS_PKG_TMP="${SIMICS_PKGS}/tmp"
export INSTALL_DIR="${SIMICS_PKGS}/simics-5-install"
export SIMICS_WORKSPACE_DIR="${simics_root}/simics-workspace"

export BKC_PKGS="${bkc_root}/Simics/Simics_Package"
export Coral_PKGS="${bkc_root}/ICE_Simulator/Coral"
export SIMICS_Script="${bkc_root}/Simics/Gold_Script"
export HOST_BIOS="${bkc_root}/PEP/Host_BIOS"
export HOST_OS="${bkc_root}/PEP/Host_OS"
export HOST_Payload="${bkc_root}/PEP/Host_Payload"
export SPH_IFWI_EP="${bkc_root}/PEP/SPH_IFWI_EP"
export SPH_IFWI_SA="${bkc_root}/SA/SPH_IFWI_SA"
export SPH_OS_PEP="${bkc_root}/PEP/SPH_OS_PEP"
export SPH_OS_SA="${bkc_root}/SA/SPH_OS_SA"


################################################################
#                      Install Simics                          #
################################################################
if [ $target == "install" ]; then
	echo_blue "Installing Simics"

	if [ ! -z "$sph_image" ]; then
		if [ ! -d ${sph_image} ]; then
        		false | ret_check "SPH image dir not found"
		fi	
	fi
	if [ ! -z "$whitley_image" ]; then
		if [ ! -d ${whitley_image} ]; then
			false | ret_check "whitley image dir not found"
		fi
	fi
        if [ ! -z "$coral_pakage" ]; then
                if [ ! -d ${coral_pakage} ]; then
                        false | ret_check "coral pakage dir not found"
                fi
        fi

	#echo_yellow "================================"
	#echo_yellow "SPH image: $sph_image"
	#echo_yellow "Boot mode: $boot_mode"
	#echo_yellow "FS mode: $fs_mode"
	#echo_yellow "coral pakage: $coral_pakage"
	#echo_yellow "Whitley image: ${whitley_image,,}"
	#echo_yellow "Rgmii Enable: ${rgmii_en,,}"
	#echo_yellow "SPH (stand-alone) only: ${sph_stand_alone,,}"
	#echo_yellow "================================"
	#echo " "
	
	mkdir -p $simics_root
	mkdir -p $SIMICS_DIR
	mkdir -p $SIMICS_PKGS
	mkdir -p $SIMICS_IMAGE
	#mkdir -p $SIMICS_PKG_TMP
	mkdir -p $INSTALL_DIR
	mkdir -p $SIMICS_WORKSPACE_DIR

	echo_blue "Simics packages take from: $SIMICS_PKGS"
	#echo_blue "Active Simics not updating automatically"
	
	# Start download Simics pkg ans install.
	echo_blue " "
	#download_head_version
        Install_Simics

	if [ -d ${SIMICS_PKGS} ]; then
		cd ${SIMICS_PKGS}
		SIMICS_BASE=`ls -1tr | grep simics-pkg-1000 | tail -1`
		SIMICS_ECLIPSE=`ls -1tr | grep simics-pkg-1001 | tail -1`
		ECLIPSE_UEFI=`ls -1tr | grep simics-pkg-7031 | tail -1`
		SIMICS_ICE_LAKE=`ls -1tr | grep simics-pkg-7047 | tail -1`
		if [ "${sph_stand_alone,,}" = false ]; then
			SIMICS_WHITLEY=`ls -1tr | grep simics-pkg-7127 | tail -1`
		fi
	else
		false | ret_check "Simics packages directory not found"
	fi

	cd ${SIMICS_DIR}
	icl_ver=`echo active_simics_icl | grep -oE "[0-9]+$"`

	active_simics_icl_lake=`ls -1 . | grep icl`
	active_simics_base=`ls -1 . | grep simics-5.`
	active_simics_uefi=`ls -1 . | grep uefi`
	active_simics_eclipse=`ls -1 . | grep eclipse-5`
	if [ "${sph_stand_alone,,}" = false ]; then
		active_simics_whitley=`ls -1 . | grep whitley`
	fi


	if [ "${sph_stand_alone,,}" = false ]; then
		echo_yellow "Active simics whitley  : $active_simics_whitley"
	fi
	echo_yellow "Active simics icl-lake : $active_simics_icl_lake"
	echo_yellow "Active simics base     : $active_simics_base"
	echo_yellow "Active simics eclipse  : $active_simics_eclipse"
	echo_yellow "Active simics uefi     : $active_simics_uefi"

	ln -s $active_simics_base active-simics-base
	ln -s $active_simics_uefi active-simics-uefi
	#ln -s $active_simics_eclipse active_simics_eclipse
	ln -s $active_simics_icl_lake active-simics-icl_lake
	#ln -s $active_simics_wb_icl_lake active-simics-wb-icl-lake
	
	if [ "${sph_stand_alone,,}" = false ]; then
		ln -s $active_simics_whitley active_simics_whitley
	fi


	echo ' '
	echo_blue "####  Updating Simics config file  ####"
	echo ' '

	echo_blue "Setting up Eclipse"
	cd ${SIMICS_DIR}; ./$active_simics_base/bin/addon-manager -b -s ./$active_simics_eclipse

	echo_blue "Setting up uefi"
	cd ${SIMICS_DIR}; ./$active_simics_base/bin/addon-manager -b -s ./$active_simics_uefi

	if [ "${sph_stand_alone,,}" = false ]; then
		# whitley installation
		echo_blue "Setting up whitley"
		cd ${SIMICS_DIR}; ./$active_simics_base/bin/addon-manager -b -s ./$active_simics_whitley
	fi
	echo_blue "Setting up icl-lake"
	cd ${SIMICS_DIR}; ./$active_simics_base/bin/addon-manager -b -s ./$active_simics_icl_lake
	
	echo_blue "Preparing simics run script"
	cp -rf $SIMICS_Script/*.simics ${SIMICS_WORKSPACE_DIR}
	echo_green "Simics run script was created at: ${SIMICS_WORKSPACE_DIR}"

        echo_blue "Setting Simics workspace in ${SIMICS_WORKSPACE_DIR}"
        mkdir -p ${SIMICS_WORKSPACE_DIR}
        cd ${SIMICS_DIR}; ./$active_simics_base/bin/project-setup ${SIMICS_WORKSPACE_DIR} --ignore-existing-files
	echo_yellow "================================"
	echo_yellow "Simics version:"
	cd ${SIMICS_WORKSPACE_DIR}; ./simics -version
	ret_check "Failed to setup simics"
	echo_yellow "================================"

	echo_green "Cleaning up: removing ${INSTALL_DIR}"
	rm -fr ${INSTALL_DIR}
	echo " "

	# setting licenses.
	lic_file="${SIMICS_DIR}/${active_simics_base}/licenses/server.lic"
	echo_green "Setting up license server at: $lic_file"
	echo "SERVER 28030@simics02p.elic.intel.com ANY" > $lic_file
	echo "USE_SERVER" >> $lic_file
	echo " "
	echo_green "Simics was successfully setup"
fi

################################################################
#                   deploy Coral package                    #
################################################################
mkdir ${install_root}/SPH_BKC/release_artifacts
tar -zxvf ${Coral_PKGS}/*.tar.gz -C ${install_root}/SPH_BKC/release_artifacts/

################################################################
#                deploy card payload and host payload       #
################################################################
if [ $card_payload_package = "e2e" ]; then
 	echo_blue "Copy HOST OS packages to ${SIMICS_IMAGE}"
        cp -rf $HOST_OS/* $SIMICS_IMAGE
        echo_blue "Copy HOST IFWI/BIOS packages to ${SIMICS_IMAGE}"
	cp -rf $HOST_BIOS/* $SIMICS_IMAGE
	echo_blue "Copy HOST Payload packages to ${SIMICS_IMAGE}"
        cp -rf ${HOST_Payload}/* ${install_root}/SPH_BKC
	echo_blue "Copy SPH IFWI packages to ${SIMICS_IMAGE}"
        cp -rf $SPH_IFWI_EP/* $SIMICS_IMAGE
	echo_blue "Copy SPH OS packages to ${SIMICS_IMAGE}"
  	tar zxvf ${SPH_OS_PEP}/*.tar.gz -C ${SIMICS_IMAGE}/
	cp -rf ${SIMICS_IMAGE}/intel/sph/OS_Image/*.img ${SIMICS_IMAGE}/disk.img
elif [ $card_payload_package = "sa" ]; then
	echo_blue "Copy SPH IFWI packages to ${SIMICS_IMAGE}"
	cp -rf $SPH_IFWI_SA/* $SIMICS_IMAGE
	echo_blue "Copy SPH OS packages to ${SIMICS_IMAGE}"
	tar zxvf ${SPH_OS_SA}/*.tar.gz -C ${SIMICS_IMAGE}/
	cp -rf ${SIMICS_IMAGE}/intel/sph/OS_Image/*.img ${SIMICS_IMAGE}/disk.img
fi

cd ${SIMICS_IMAGE}
ls
#rm ./intel -rf
#rm *.tar.gz -rf

################################################################
#                        Clean Simics                          #
################################################################
if [ $target = "clean" ]; then
        echo_blue "Removing Simics"
        rm -fr ${install_root}/SPH_BKC

fi


# python
#cmd_1s1i = './Alibab_script_1S-1Ins_v6.sh %s 2>&1 | tee ./log_data/mxnet1s1ns_%s.log' % (omp_threads, omp_threads)