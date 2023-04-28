#include <opa_common.h>

#include <clocale>
#include <codecvt>
#include <cwchar>
#include <locale>

#include <Python.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>

void run_python_entry(std::string args) {
  gzmsg << "On thread " << args << std::endl;

  Py_Initialize();
  int res = PyRun_SimpleString("import chdrft.sim.gz.gz_entrypoint");
  std::string params =  R"(chdrft.sim.gz.gz_entrypoint.enter(""")" + args + R"( """ ) )";
  PyRun_SimpleString(params .data());
  gzmsg << "Result of run: " << res << std::endl;
  Py_Finalize();
}

void run_python(std::string args) {
  run_python_entry(args);
  // std::thread thread(run_python_entry, args);
  // thread.detach();
}

class SampleSystem : public gz::sim::System,
                     public gz::sim::ISystemConfigure,
                     public gz::sim::ISystemPostUpdate {
public:
  SampleSystem() {}

public:
  void Configure(const gz::sim::Entity &_entity, const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &_eventMgr) override {
    gzmsg << "Configure" << _sdf->ToString("MyEntity") << std::endl;
    auto e = _sdf->Get<std::string>("args");
    gzmsg << "ARGS" << e << std::endl;
    run_python(e);
  }

public:
  ~SampleSystem() override {}
  int i = 0;

public:
  void PostUpdate(const gz::sim::UpdateInfo &_info,
                  const gz::sim::EntityComponentManager &_ecm) override {

    // This is a simple example of how to get information from UpdateInfo.
    std::string msg = "Hello, world! Simulation is ";
    if (!_info.paused) msg += "not ";
    msg += "paused.";

    // Messages printed with gzmsg only show when running with verbosity 3 or
    // higher (i.e. gz sim -v 3)
    // gzmsg << msg << std::endl;
  }
};

GZ_ADD_PLUGIN(SampleSystem, gz::sim::System, SampleSystem::ISystemPostUpdate,
              SampleSystem::ISystemConfigure)

int abc() { return 0; }
