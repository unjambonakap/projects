package main;

import (
  "fmt"
  "os"
  "chdrft/cel-server/app"
)

type FuncTest[K any] func (chdrft_cel.CelEvaluator, string) (K, error);

func do_test[K any](expr string, data string, f FuncTest[K]) K{
  a, err := chdrft_cel.CreateEvaluator(expr);

  fmt.Printf("Starting test >> %v %v\n", expr, data)
  if err != nil {
      fmt.Fprintf(os.Stderr, "error: %v\n", err)
      os.Exit(1)
  }
  res, err := f(*a, data);
  
  if err != nil {
      fmt.Fprintf(os.Stderr, "error: %v\n", err)
      os.Exit(1)
  }
  fmt.Printf("Got >> %v\n", res);
  return  res


}

func main(){
  fmt.Println("Hello");
  do_test(`[obj.test, obj.test+ "123"]`, `{ "test": "22"}`, chdrft_cel.CelEvaluator.EvaluateToStringList)
  do_test(`obj.test == 22`, `{ "test": "22"}`, chdrft_cel.CelEvaluator.EvaluateToBool)
  do_test(`obj.test == 22`, `{ "test": 22}`, chdrft_cel.CelEvaluator.EvaluateToBool)
  a := do_test(`cel_server.base.PII{a: 3, b:  34}`, `{ "test": 22}`, chdrft_cel.CelEvaluator.EvaluateToPII);
  fmt.Println("FUU %v %v\n", a.GetA(), a.GetB());

  //do_test(`{"data": obj.test}`, `{ "test": {"xo": 123}}`, chdrft_cel.CelEvaluator.EvaluateToJson)


}
