package chdrft_cel

import (
  "errors"
  "fmt"

  self_proto "chdrft/cel-server/app/proto"
  "reflect"
  "google.golang.org/protobuf/encoding/protojson"
  structpb "github.com/golang/protobuf/ptypes/struct"
  "github.com/google/cel-go/cel"
  "github.com/google/cel-go/checker/decls"
  "github.com/google/cel-go/common/types/ref"
  "github.com/google/cel-go/common/types/pb"
  "google.golang.org/protobuf/proto"
)

type CelEvaluator struct {
  program cel.Program 
}

func CreateEvaluator(filter string) (*CelEvaluator, error){
  ds := cel.Declarations(
    decls.NewIdent("obj", decls.NewMapType(decls.String, decls.Dyn), nil),
  )
  pb.DefaultDb.RegisterMessage(&self_proto.PII{})
  //x, y := pb.DefaultDb.DescribeType(string((&self_proto.PII{}).ProtoReflect().Descriptor().FullName()))
	//fmt.Printf("GOT >> %v %v\n", x, y)
  //fmt.Printf("FUU %v\n", self_proto.File_proto_base_proto.Messages().Get(0));

  env, err := cel.NewEnv(ds)
  if err != nil {
    return nil, err;
  }

  prs, iss := env.Parse(filter)
  if iss != nil && iss.Err() != nil {
    return nil, iss.Err();
  }

  chk, iss := env.Check(prs)
  if iss != nil && iss.Err() != nil {
    return nil, iss.Err();
  }

  prg, err := env.Program(chk)
  if err != nil {
    return nil, err
  }
  return &CelEvaluator{prg}, nil; 
}

func (this *CelEvaluator) GetValue(json_data string) (ref.Val, error) {
  var spb structpb.Struct
  if err := protojson.Unmarshal([]byte(json_data), &spb); err != nil {
    return nil, err
  }

  val, _, err := this.program.Eval(map[string]interface{}{"obj": &spb})
  if err != nil {
    fmt.Printf("Eval error %v\n", err);
    return nil, err
  }
  return val, nil;


}

func  EvaluateToMessage[K any](this *CelEvaluator, json_data string) (K, error) {
  var r0 K
  val, err := this.GetValue(json_data)
  if err != nil {
    return r0, err;
  }

  res, err := val.ConvertToNative(reflect.TypeFor[*K]())
  if err != nil{
    return r0, err;
  }

  if rx, ok := res.(*K) ; ok {
    return *rx, nil
  }

  return r0, errors.New("Failed to convert - weird")
}

func  EvaluateTo[K any ](this *CelEvaluator, json_data string) (K, error) {
  var result K
  val, err := this.GetValue(json_data)
  if err != nil {
    return result, err;
  }

  //fmt.Printf("Got .> %v\n", reflect.TypeOf(val.Value()));
  result, ok := val.Value().(K)
  if  ok {
    return result, nil
  }

  res, err := val.ConvertToNative(reflect.TypeOf(result))
  if err != nil{
    return result, err;
  }

  result, ok = res.(K)
  if  !ok {
    return result, errors.New("Failed to convert");
  }
  return result, nil;
}


func (this CelEvaluator) EvaluateToJson(json_data string) ([]byte, error) {
  val, err := EvaluateTo[proto.Message](&this, json_data)
  if err != nil {
    return nil, err
  }

  res, err :=  protojson.Marshal(val)
  return res, err
}

func (this CelEvaluator) EvaluateToBool(json_data string) (bool, error) {
  return EvaluateTo[bool](&this, json_data)
}

func (this CelEvaluator) EvaluateToStringList(json_data string) ([]string, error) {
  return EvaluateTo[[]string](&this, json_data)
}

func (this CelEvaluator) EvaluateToInt(json_data string) (int, error) {
  return EvaluateTo[int](&this, json_data)
}


func (this CelEvaluator) EvaluateToString(json_data string) (string, error) {
  return EvaluateTo[string](&this, json_data)
}

func (this CelEvaluator) EvaluateToPII(json_data string) (self_proto.PII, error) {
  return EvaluateToMessage[self_proto.PII](&this, json_data)
}
