#!/bin/bash

set -e

echo "🚀 Iniciando instalación de entorno Android para reco_trading..."

# ==============================

# ACTUALIZAR SISTEMA

# ==============================

echo "📦 Actualizando paquetes..."
sudo apt update

# ==============================

# DEPENDENCIAS

# ==============================

echo "🔧 Instalando dependencias necesarias..."
sudo apt install -y 
software-properties-common 
build-essential 
libssl-dev 
zlib1g-dev 
libncurses5-dev 
libffi-dev 
libsqlite3-dev 
libreadline-dev 
libbz2-dev 
wget 
curl 
llvm 
git 
zip 
unzip 
openjdk-17-jdk

# ==============================

# DESCARGAR PYTHON 3.10

# ==============================

echo "🐍 Descargando Python 3.10..."

cd /tmp

if [ ! -f Python-3.10.13.tgz ]; then
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
fi

tar -xf Python-3.10.13.tgz
cd Python-3.10.13

# ==============================

# COMPILAR PYTHON

# ==============================

echo "⚙️ Compilando Python 3.10 (esto puede tardar)..."

./configure --enable-optimizations
make -j$(nproc)

echo "📥 Instalando Python 3.10..."
sudo make altinstall

# ==============================

# VERIFICAR

# ==============================

echo "✅ Verificando instalación..."
python3.10 --version

# ==============================

# CREAR ENTORNO VIRTUAL

# ==============================

echo "🧪 Creando entorno virtual..."

cd $HOME

if [ ! -d "venv_android" ]; then
python3.10 -m venv venv_android
fi

source venv_android/bin/activate

# ==============================

# ACTUALIZAR PIP

# ==============================

echo "⬆️ Actualizando pip..."
pip install --upgrade pip

# ==============================

# INSTALAR DEPENDENCIAS ANDROID

# ==============================

echo "📲 Instalando Buildozer + Kivy..."

pip install cython
pip install buildozer
pip install kivy
pip install requests

# ==============================

# FINAL

# ==============================

echo ""
echo "🎉 ENTORNO LISTO!"
echo ""
echo "👉 Activa el entorno con:"
echo "source ~/venv_android/bin/activate"
echo ""
echo "👉 Luego ve a tu app:"
echo "cd app_android"
echo ""
echo "👉 Compila con:"
echo "buildozer -v android debug"
echo ""
echo "🔥 Todo listo para crear tu APK"
