cd /home/benoit/repos/icsharp
./build.sh
./pack.sh 1.0.8

#cd /home/benoit/work/links/Kerbal Space Program/GameData/kRPC
cd /home/benoit/repos/krpc
pushd lib/extra
./info.sh
popd

bazel build -s //server --spawn_strategy=standalone
./tools/install.sh
