#include "opa/utils/map_util.h"
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <opa/utils/base.h>
#include <opa/utils/buffer_reader.h>
#include <opa/utils/buffer_writer.h>
#include <opa/utils/range.h>
#include <opa_common.h>

#include <nlohmann/json.hpp>
__BEGIN_DECLS
#pragma warning(push, 0)
#include <gpmf-parser/GPMF_parser.h>
#include <gpmf-parser/GPMF_utils.h>
#include <gpmf-parser/demo/GPMF_mp4reader.h>
#pragma warning(pop)
__END_DECLS

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>

using namespace std;

using namespace opa;
using namespace opa::utils;
using namespace std;

typedef void (*action_func)();
std::unordered_map<string, action_func> actions;

DEFINE_string(action, "", "");
DEFINE_string(outfile, "", "");
DEFINE_string(infile, "", "");
DEFINE_string(opt_type, "", "");
DEFINE_int32(max_payload, -1, "");
DEFINE_double(huber_loss, -1, "");
DEFINE_int32(start_time_sec, -1, "");
DEFINE_int32(end_time_sec, -1, "");

#define GPMF_OP(name, ...) OPA_CHECK0(GPMF_##name(__VA_ARGS__) == GPMF_OK)

struct MoovPayloadDesc {
  size_t offset;
  size_t timescale;
  size_t duration;
  size_t sz;
  int track_id;
  size_t cts;
  size_t dts;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(MoovPayloadDesc, offset, timescale, duration, sz, track_id, cts,
                                 dts)
};

struct Info {
  std::string filename;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Info, filename)
};

struct Document {
  Info info;
  std::vector<MoovPayloadDesc> payloads;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Document, info, payloads)
};

struct SampleValue {
  double value;
  int unit;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SampleValue, value, unit)
};

struct Sample {
  std::string key;
  std::vector<SampleValue> values;
  int payload_id;
  int inner_payload_pos;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Sample, key, values, payload_id, inner_payload_pos)
};

struct Payload {
  double duration;
  double start_time;
  int payload_id;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Payload, duration, start_time, payload_id)
};

struct UnitDesc {
  int unit_id;
  std::string desc;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(UnitDesc, unit_id, desc)
};

struct Result {
  std::map<std::string, std::vector<Sample> > code2samples;
  std::vector<Payload> payloads;
  std::vector<UnitDesc> units;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Result, code2samples, payloads, units)
};

struct Processor {
  static constexpr auto recurse_tolerant = (GPMF_LEVELS)(GPMF_RECURSE_LEVELS | GPMF_TOLERANT);
  static constexpr auto current_tolerant = (GPMF_LEVELS)(GPMF_CURRENT_LEVEL | GPMF_TOLERANT);
  Result res;

  std::map<std::string, UnitDesc> units_map;
  UnitDesc get_unit(const std::string &s) {
    if (!units_map.count(s)) {
      units_map[s] = UnitDesc{ .unit_id = units_map.size(), .desc = s };
    }
    return units_map[s];
  }

  std::pair<bool, std::vector<UnitDesc> > get_units(GPMF_stream &stream) {

#define MAX_UNITS 64
#define MAX_UNITLEN 8
    char units[MAX_UNITS][MAX_UNITLEN] = { "" };
    uint32_t unit_samples = 1;

    char complextype[MAX_UNITS] = { "" };
    uint32_t type_samples = 1;

    uint32_t i, j;
    GPMF_stream find_stream;
    std::vector<UnitDesc> res;

    std::map<char, char> remap_char = {
      { 0xb0, '0' },
      { 0xb2, '2' },
      { 0xb3, '3' },
      { 0xb5, 'u' },
    }; // 0xB0 (°), 0xB2 (²), 0xB3 (³) and 0xB5

    // Search for any units to display
    GPMF_CopyState(&stream, &find_stream);
    if (GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_SI_UNITS, current_tolerant) ||
        GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_UNITS, current_tolerant)) {
      char *data = (char *)GPMF_RawData(&find_stream);
      uint32_t ssize = GPMF_StructSize(&find_stream);
      if (ssize > MAX_UNITLEN - 1) ssize = MAX_UNITLEN - 1;
      unit_samples = GPMF_Repeat(&find_stream);

      for (i = 0; i < unit_samples && i < MAX_UNITS; i++) {
        memcpy(units[i], data, ssize);
        units[i][ssize] = 0;
        REP (j, ssize)
          units[i][j] = glib::gtl::FindWithDefault(remap_char, units[i][j], units[i][j]);
        res.pb(get_unit(std::string(units[i])));
        data += ssize;
      }
    }

    // Search for TYPE if Complex
    GPMF_CopyState(&stream, &find_stream);
    type_samples = 0;
    if (GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_TYPE, current_tolerant)) {
      char *data = (char *)GPMF_RawData(&find_stream);
      uint32_t ssize = GPMF_StructSize(&find_stream);
      if (ssize > MAX_UNITLEN - 1) ssize = MAX_UNITLEN - 1;
      type_samples = GPMF_Repeat(&find_stream);

      for (i = 0; i < type_samples && i < MAX_UNITS; i++) {
        complextype[i] = data[i];
      }
      return { false, {} };
    }
    return { true, res };
  }

  void process_stream(GPMF_stream &stream, const MoovPayloadDesc &payloadDesc,
                      const Payload &payload) {
    GPMF_ERR ret;
    ret = GPMF_SeekToSamples(&stream);
    if (GPMF_OK != ret) return;

    uint32_t key = GPMF_Key(&stream);
    GPMF_SampleType type = GPMF_Type(&stream);
    uint32_t samples = GPMF_Repeat(&stream);

    if (!samples) return;
    std::string skey{ PRINTF_4CC(key) };

    OPA_DISP("Proc ", skey);
    auto [ok, units] = get_units(stream);
    if (!ok) {
      printf("Ignoring %s because not supported units\n", skey.data());
      return;
    }

    char *rawdata = (char *)GPMF_RawData(&stream);
    uint32_t elements = GPMF_ElementsInStruct(&stream);
    uint32_t buffersize = samples * elements * sizeof(double);
    std::vector<double> tmp(samples * elements);
    int tmp_pos = 0;
    // GPMF_FormattedData(ms, tmpbuffer, buffersize, 0, samples); // Output data in LittleEnd,
    // but no scale
    if (GPMF_OK == GPMF_ScaledData(&stream, &tmp[0], tmp.size() * sizeof(tmp[0]), 0, samples,
                                   GPMF_TYPE_DOUBLE)) // Output scaled data as floats
    {

      int pos = 0;
      REP (i, samples) {
        OPA_CHECK0(type != GPMF_TYPE_STRING_ASCII);
        auto sample = Sample{
          .key = skey,
          .values = STD_RANGE(0, elements) |
                    STD_TSFX((SampleValue{
                      .value = tmp[i * elements + x],
                      .unit = units.size() == 0 ? -1 : units[x % units.size()].unit_id,
                    })) |
                    STD_VEC,
          .payload_id = payload.payload_id,
          .inner_payload_pos = i,
        };
        res.code2samples[skey].pb(sample);
        tmp_pos += elements;
      }
    }
  }

  Result process(const std::string &filename) {
    res = Result{};
    Document doc = nlohmann::json::parse(read_file(filename)).template get<Document>();
    auto content = read_file(doc.info.filename);

    int pid = 0;
    for (auto &pdesc : doc.payloads) {
      GPMF_stream stream = { 0 };
      GPMF_OP(Init, &stream, (u32 *)&content.data()[pdesc.offset], pdesc.sz);
      auto payload = Payload{ .duration = (double)pdesc.duration / pdesc.timescale,
                              .start_time = (double)pdesc.cts / pdesc.timescale,
                              .payload_id = pid++ };
      if (FLAGS_start_time_sec != -1 &&
          payload.start_time + payload.duration < FLAGS_start_time_sec)
        continue;
      if (FLAGS_end_time_sec != -1 && payload.start_time > FLAGS_end_time_sec) continue;
      res.payloads.pb(payload);

      while (GPMF_OK == GPMF_FindNext(&stream, STR2FOURCC("STRM"), recurse_tolerant)) {
        process_stream(stream, pdesc, payload);
      }
      GPMF_OP(Free, &stream);
      if (FLAGS_max_payload != -1 && pid >= FLAGS_max_payload) break;
    }
    res.units = units_map | LQ::values | STD_VEC;
    std::sort(ALL(res.units), [](auto a, auto b) { return a.unit_id < b.unit_id; });
    return res;
  }
};

nlohmann::json convert(const std::string &filename) {
  Processor proc;
  return proc.process(filename);
}
template <class T> void dump(const T &a) {
  auto outjs = nlohmann::json(a);

  if (FLAGS_outfile != "") {
    std::ofstream ofs(FLAGS_outfile);
    ofs << std::setw(4) << outjs << std::endl;
  } else {
    std::cout << std::setw(4) << outjs << std::endl;
  }
}

void convert_gpmf() {
  auto out = convert(FLAGS_infile);
  dump(out);
}

struct OptData {
  std::vector<std::array<double, 3> > meas;
  std::vector<std::array<double, 3> > real;
  std::vector<std::array<double, 4> > cori;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(OptData, meas, real, cori)
};

class ErrTerm {
public:
  ErrTerm(const Eigen::Vector3d &obs, const Eigen::Vector3d &real,
          const Eigen::Matrix<double, 3, 3> &sqrt_information)
      : obs(obs), real(real), sqrt_information_(std::move(sqrt_information)) {}

  template <typename T> bool operator()(const T *const q_pose_ptr, T *residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T> > q_pose(q_pose_ptr);

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> est = q_pose * real.template cast<T>();

    // Compute the error between the two orientation estimates.

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals = (obs - est);

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &obs, const Eigen::Vector3d &real,
                                     const Eigen::Matrix<double, 3, 3> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<ErrTerm, 3, 4>(new ErrTerm(obs, real, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector3d obs, real;
  const Eigen::Matrix<double, 3, 3> sqrt_information_;
};

class Opt2 {
public:
  Opt2(const Eigen::Quaterniond &cori, const Eigen::Matrix<double, 3, 3> &sqrt_information)
      : cori(cori), sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *fix_vec_world, const T *fix_vec, T *residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T> > fv_world(fix_vec_world);
    Eigen::Map<const Eigen::Quaternion<T> > fv(fix_vec);

    std::array<double, 3> arr = { 0, 0, 1 };
    Eigen::Map<const Eigen::Vector3d> pv(arr.data());
    Eigen::Map<Eigen::Vector3<T> > residuals(residuals_ptr);
    residuals =
      cori.template cast<T>() * fv_world * pv.template cast<T>() - fv * pv.template cast<T>();

    //// Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Quaterniond &cori,
                                     const Eigen::Matrix<double, 3, 3> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<Opt2, 3, 4, 4>(new Opt2(cori, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Quaterniond cori;
  const Eigen::Matrix<double, 3, 3> sqrt_information_;
};

struct OptResult {
  std::array<double, 4> fv;
  std::array<double, 4> fv_world;
  std::array<double, 4> q_pose;
  bool converged = false;
  double cost_initial;
  double cost_final;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(OptResult, fv, fv_world, q_pose, converged, cost_initial,
                                 cost_final)
};

void do_opt1() {

  OptData data = nlohmann::json::parse(read_file(FLAGS_infile)).template get<OptData>();
  ceres::Problem problem;
  ceres::LossFunction *loss_function =
    FLAGS_huber_loss < 0 ? nullptr : new ceres::HuberLoss(FLAGS_huber_loss);
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;

  std::array<double, 4> fv = { 0, 0, 0, 1 };       // xyzw
  std::array<double, 4> fv_world = { 0, 0, 0, 1 }; // xyzw
  std::array<double, 4> q_pose = { 0, 0, 0, 1 };   // xyzw
  const Eigen::Matrix<double, 3, 3> sqrt_information = Eigen::Matrix<double, 3, 3>::Identity();
  if (FLAGS_opt_type == "cori") {

    REP (i, data.meas.size()) {
      ceres::CostFunction *cost_function =
        ErrTerm::Create(Eigen::Map<Eigen::Vector3d>(data.meas[i].data()),
                        Eigen::Map<Eigen::Vector3d>(data.real[i].data()), sqrt_information);
      problem.AddResidualBlock(cost_function, loss_function, q_pose.data());
    }
    problem.SetManifold(q_pose.data(), quaternion_manifold);
  } else {

    REP (i, data.meas.size()) {
      ceres::CostFunction *cost_function =
        Opt2::Create(Eigen::Map<Eigen::Quaterniond>(data.cori[i].data()), sqrt_information);
      problem.AddResidualBlock(cost_function, loss_function, fv_world.data(), fv.data());
    }
    problem.SetManifold(fv_world.data(), quaternion_manifold);
    problem.SetManifold(fv.data(), quaternion_manifold);
  }
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  auto res = OptResult{
    .fv = fv,
    .fv_world = fv_world,
    .q_pose = q_pose,
    .converged = summary.termination_type == ceres::TerminationType::CONVERGENCE,
    .cost_initial = summary.initial_cost,
    .cost_final = summary.final_cost,
  };
  std::cout << summary.FullReport() << '\n';
  dump(res);
}

int main(int argc, char *argv[]) {
  opa::init::opa_init(argc, argv);
  actions["convert_gpmf"] = convert_gpmf;
  actions["do_opt1"] = do_opt1;

  string action = FLAGS_action;
  OPA_CHECK0(!action.empty());
  OPA_CHECK0(actions.count(action));
  actions[action]();
  return 0;
}
