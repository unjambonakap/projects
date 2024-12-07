package test

import (
  "errors"
	"encoding/json"
	"testing"

	"google.golang.org/protobuf/encoding/protojson"
	structpb "github.com/golang/protobuf/ptypes/struct"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
)

type Payload struct {
	Strs []string          `json:"strs"`
	Data map[string]string `json:"data"`
}

type MyStruct struct {
	Num     int64   `json:"num"`
	Str     string  `json:"str"`
	Payload Payload `json:"payload"`
}
type CelEvaluator struct {
  program cel.Program 
}

func createEvaluator(filter string) (*CelEvaluator, error){
			ds := cel.Declarations(
				decls.NewIdent("obj", decls.NewMapType(decls.String, decls.Dyn), nil),
			)

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

func (this *CelEvaluator) evaluate(json_data string) (bool, error) {
  var spb structpb.Struct
  if err := protojson.Unmarshal([]byte(json_data), &spb); err != nil {
    return false, err
  }

    val, _, err := this.program.Eval(map[string]interface{}{"obj": &spb})
    if err != nil {
      return false, err
    }

    gotMatch, ok := val.Value().(bool)
    if !ok {
      return false, errors.New("Failed to convert to bool");
    }
    return gotMatch, nil
}

func TestCELStructJSON(t *testing.T) {
  for _, tc := range []struct {
    name      string
    filter    string
    myStruct  *MyStruct
    wantMatch bool
  }{{
    name:      "simple match",
    filter:    `myStruct.str == "hello" && "world" in myStruct.payload.data`,
    myStruct:  &MyStruct{Num: 10, Str: "hello", Payload: Payload{Data: map[string]string{"world": "foobar"}}},
    wantMatch: true,
  }, {
    name:      "simple mismatch",
    filter:    `myStruct.num > 9000 && "banana" in myStruct.payload.strs`,
    myStruct:  &MyStruct{Num: 9001, Str: "blah", Payload: Payload{Strs: []string{"kiwi", "orange"}, Data: map[string]string{"mars": "goober"}}},
    wantMatch: false,
  }} {
    t.Run(tc.name, func(t *testing.T) {
      // First build the CEL program.
      ds := cel.Declarations(
        decls.NewIdent("myStruct", decls.NewMapType(decls.String, decls.Dyn), nil),
      )

      env, err := cel.NewEnv(ds)
      if err != nil {
        t.Fatal(err)
      }

      prs, iss := env.Parse(tc.filter)
      if iss != nil && iss.Err() != nil {
        t.Fatal(iss.Err())
      }

      chk, iss := env.Check(prs)
      if iss != nil && iss.Err() != nil {
        t.Fatal(iss.Err())
      }

      prg, err := env.Program(chk)
      if err != nil {
        t.Fatal(err)
      }

      // Now, get the input in the correct format (conversion: Go struct -> JSON -> structpb).
      j, err := json.Marshal(tc.myStruct)
      if err != nil {
        t.Fatal(err)
      }

      var spb structpb.Struct
      if err := protojson.Unmarshal(j, &spb); err != nil {
        t.Fatal(err)
      }

      // Now, evaluate the program and check the output.
      val, _, err := prg.Eval(map[string]interface{}{"myStruct": &spb})
      if err != nil {
        t.Fatal(err)
      }

      gotMatch, ok := val.Value().(bool)
      if !ok {
        t.Fatalf("failed to convert %+v to bool", val)
      }

      if gotMatch != tc.wantMatch {
        t.Errorf("expected cel(%q, %s) to be %v", tc.filter, string(j), tc.wantMatch)
      }
    })
  }
}
