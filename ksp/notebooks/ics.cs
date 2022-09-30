using System.Reflection;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;
using KRPC.SpaceCenter.Services;


var y =new [] { 123, 456};
Console.WriteLine($"FUUU >> y.ToList()2")

Console.WriteLine(KRPC.Kernel.Instance.Current == null);

//_env.AddSearchPath("/home/benoit/.local/share/Steam/steamapps/common/Kerbal Space Program/KSP_Data/Managed");
//_env.AddSearchPath("/home/benoit/.local/share/Steam/steamapps/common/Kerbal Space Program/GameData/kRPC");
//_env.AddReference("Assembly-CSharp.dll");
//_env.AddReference("KRPC.dll");
//var x= (KRPC.Kernel)_env.ExtraParams.Data;

Console.WriteLine(typeof(KRPC.Kernel).Assembly.Location);
Console.WriteLine(typeof(KRPC.Addon).Assembly.Location);
Console.WriteLine(KRPC.Kernel.Instance ==null);
Console.WriteLine(KRPC.Addon.Instance ==null);

Console.WriteLine(ShipConstruction.Instance);


int recoverCount = 0;
string path;
ShipTemplate template;
ShipConstruct construct;
string config = null;
List<string> data = new List<string>();
//path =ShipConstruction.GetSavePath("Auto-Saved Ship");
template =ShipConstruction.LoadTemplate("./Ships/VAB/SpaceX Falcon Heavy.craft");
template =ShipConstruction.LoadTemplate("./Ships/VAB/SpaceX Falcon 9 Block 5.craft");
data.Add(template.config == null ? "Null" : "Ok");
config = template.config.ToString();
path += "1";
//return;
var launchSite = "VAB";
var flightState =HighLogic.CurrentGame.flightState;

//var vesselsToRecover = ShipConstruction.FindVesselsLandedAt(, launchSite);
foreach (var v in flightState.protoVessels){
    recoverCount += 1;
    ShipConstruction.RecoverVesselFromFlight(v, flightState);
}

EditorLogic.fetch.launchVessel(launchSite);
Console.WriteLine($"PATH >> {path}");
//TimeWarp.SetRate(1, true);
//TimeWarp.fetch.physicsWarpRates[0] = 0.1f;
//TimeWarp.SetRate(0, true);

TimeWarp.SetRate(1, true);
TimeWarp.fetch.physicsWarpRates[0] = 0.1f;
TimeWarp.fetch.warpRates[0] = 0.1f;
TimeWarp.SetRate(0, true);


var mun = KRPC.SpaceCenter.Services.SpaceCenter.Bodies["Mun"];
var earth = KRPC.SpaceCenter.Services.SpaceCenter.Bodies["Kerbin"];

var active = FlightGlobals.fetch.activeVessel;
for (int i = 0; i < active.Parts.Count; ++i) {
// need checks on shielded components
var p = active.Parts[i];

for (int j = 0; j < p.Modules.Count; ++j) {
var m = p.Modules[j];
var wing = m as ModuleLiftingSurface;
if (wing != null)
Console.WriteLine(p.ToString());
}
}


var vessel = KRPC.SpaceCenter.Services.SpaceCenter.ActiveVessel;
foreach (var part in vessel.InternalVessel.Parts)
{
Console.WriteLine($"{part.children.Count} {part.parent==null} {part.GetHashCode()}");
foreach(var child in part.children) Console.WriteLine(child.GetHashCode());
}

using KRPC.Service;
using KRPC.Service.Attributes;
using KRPC.SpaceCenter.ExtensionMethods;
using KRPC.SpaceCenter.ExternalAPI;
using KRPC.Utils;
using UnityEngine;


var pos = new Vector3d(0.0, 610000.0, 0.0);
var worldPos= earth.ReferenceFrame.PositionToWorldSpace(pos);
var vel = new Vector3d(0, 0, 10);
var worldVel = earth.ReferenceFrame.VelocityToWorldSpace(pos, vel);
Console.WriteLine(worldVel);
Console.WriteLine(worldPos);
Console.WriteLine(earth.InternalBody.position);
var relativeWorldVelocity = worldVel - earth.InternalBody.getRFrmVel(worldPos);
Console.WriteLine(relativeWorldVelocity);
var krpcV = KRPC.SpaceCenter.Services.SpaceCenter.ActiveVessel;
var f = krpcV.Flight(earth.ReferenceFrame);

Console.WriteLine(f.SimulateAerodynamicForceAt(earth, pos.ToTuple(), vel.ToTuple()).ToVector());
//Console.WriteLine(StockAerodynamics.SimAeroForce(earth.InternalBody, active, relativeWorldVelocity, worldPos));

    var active = FlightGlobals.fetch.activeVessel;
foreach(var part in active.parts) if (part.rb!=null) {
part.rb.angularVelocity = new Vector3(0,0,0);
part.rb.velocity = Vector3.zero;
}
//var vb = active.GetComponent<Rigidbody>();
//vb.angularVelocity = Vector3.zero;
//vb.angularDrag = 0f;
//
//active.angularMomentum = Vector3.zero;
//active.angularVelocity = Vector3.zero;
//active.angularVelocityD = Vector3d.zero;
//active.MOI = Vector3.zero;
//active.ChangeWorldVelocity(new Vector3d(100, 0, 0));
////
//

string launchSite = "VAB";
    //FlightDriver.RevertToLaunch();
    //FlightDriver.RevertToPrelaunch(EditorFacility.VAB);
    FlightDriver.ReturnToEditor(EditorFacility.VAB);
//EditorLogic.LoadShipFromFile("./Ships/VAB/SpaceX Falcon Heavy.craft");

EditorLogic.LoadShipFromFile("./Ships/VAB/SpaceX Falcon 9 Block 5.craft");

EditorLogic.fetch.launchVessel(launchSite);

               Vector3 gpsPos = WorldPositionToGeoCoords(worldPos, FlightGlobals.currentMainBody);


    //FlightDriver.SetPause(false);
//Console.WriteLine(path);

FlightDriver.RevertToLaunch()
FlightGlobals.fetch.Start()

var active = FlightGlobals.fetch.activeVessel;

Console.WriteLine(active.GetWorldPos3D());
Console.WriteLine(FlightGlobals.currentMainBody.position);
Console.WriteLine(FlightGlobals.currentMainBody.getTruePositionAtUT(Planetarium.GetUniversalTime()));
Console.WriteLine(Planetarium.GetUniversalTime());
Console.WriteLine(Planetarium.FrameIsRotating());
Console.WriteLine(Planetarium.fetch.Sun.position);
Console.WriteLine(JsonConvert.SerializeObject(KRPC.SpaceCenter.Services.SpaceCenter.Bodies.Keys));


string launchSite = "VAB";
    //FlightDriver.RevertToLaunch();
    //FlightDriver.RevertToPrelaunch(EditorFacility.VAB);
    //FlightDriver.ReturnToEditor(EditorFacility.VAB);
//EditorLogic.LoadShipFromFile("./Ships/VAB/SpaceX Falcon Heavy.craft");

EditorLogic.fetch.launchVessel(launchSite);



Console.WriteLine(active.mainBody.GetWorldSurfacePosition(active.latitude, active.longitude, active.altitude));


var active = FlightGlobals.fetch.activeVessel;
var vec = active.GetWorldPos3D() - FlightGlobals.currentMainBody.position;
active.SetPosition(active.GetWorldPos3D() - vec.normalized*2000000, true);


Console.WriteLine(active.angularMomentum);
Console.WriteLine(active.GetComponent<Rigidbody>().angularDrag);


var mun = KRPC.SpaceCenter.Services.SpaceCenter.Bodies["Mun"];
var earth = KRPC.SpaceCenter.Services.SpaceCenter.Bodies["Kerbin"];
Console.WriteLine(mun.Position(earth.ReferenceFrame));

var cur =active.GetTransform().eulerAngles;
cur.x += 10;
active.SetRotation(Quaternion.Euler(cur));
//Console.WriteLine(active.srfRelRotation.eulerAngles);

Console.WriteLine(FlightGlobals.Vessels.Count);
Console.WriteLine(FlightGlobals.fetch.activeVessel==null);

//Console.WriteLine(ShipConstruction.FindVesselsAtLaunchSite(HighLogic.CurrentGame.flightState, launchSite).Count);

foreach (var vessel in FlightGlobals.Vessels)
    ShipConstruction.RecoverVesselFromFlight(vessel.protoVessel, HighLogic.CurrentGame.flightState, true);

Console.WriteLine(FlightGlobals.ActiveVessel.GetWorldPos3D())


Console.WriteLine(FlightGlobals.ActiveVessel.GetWorldPos3D())

var craftUrl = "./Ships/VAB/SpaceX Falcon Heavy.craft";
var p =FlightGlobals.ActiveVessel.GetWorldPos3D();
p.z += 35;
VesselMover.VesselSpawn.instance.SpawnVesselFromCraftFile(craftUrl, p, 0, 0 )

var p =FlightGlobals.ActiveVessel.GetWorldPos3D();
var gpsPos = VesselMover.VesselSpawn.WorldPositionToGeoCoords(p, FlightGlobals.currentMainBody);
gpsPos.z += 35;
VesselMover.VesselSpawn.instance.SpawnVesselFromCraftFile(craftUrl, gpsPos, 0, 0 );

VesselMover.VesselSpawn.instance.SpawnCraftRoutine(craftUrl)



string launchSite="VAB";
var vesselsToRecover = ShipConstruction.FindVesselsLandedAt(HighLogic.CurrentGame.flightState, launchSite);

    foreach (var vessel in vesselsToRecover)
        ShipConstruction.RecoverVesselFromFlight(vessel, HighLogic.CurrentGame.flightState, true);




FlightGlobals.fetch.SetVesselTarget (null, true);


KRPC.SpaceCenter.Services.SpaceCenter.ClearTarget()


KRPC.SpaceCenter.Services.SpaceCenter.LaunchVesselFromVAB("SpaceX Falcon 9 Block 5", true)


FlightDriver.RevertToLaunch();

KRPC.Kernel.Instance.Push(() => {
    TimeWarp.fetch.physicsWarpRates[0] = 0.5f;
});

Console.WriteLine(y);

x.Push()

Console.WriteLine(JsonConvert.SerializeObject(KRPC.Kernel.Instance.UpdateCount));
Console.WriteLine(JsonConvert.SerializeObject(KRPC.Addon.Instance.UpdateCount));


Console.WriteLine(JsonConvert.SerializeObject(TimeWarp.fetch.Mode))


Console.WriteLine(JsonConvert.SerializeObject(TimeWarp.CurrentRateIndex))

_env.Debug = true;

try {
Console.WriteLine("A BC");
//throw new Exception("123");

}catch(Exception e){
Console.WriteLine($"GOT >> {e.Message}");
}

public class Test3 {
    public int f1(){
        return 5;
    }

    
}

Console.WriteLine(new Test3().f1());

var ass = Assembly.LoadFile("/home/benoit/.local/share/Steam/steamapps/common/Kerbal Space Program/KSP_Data/Managed/Assembly-CSharp.dll")

foreach(var x in ass.GetTypes().Select(y=>y.FullName))
Console.WriteLine(x);
