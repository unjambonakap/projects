var fs = require('fs');
var MP4Box = require('/home/benoit/repos/mp4box.js/dist/mp4box.all.js');

if (process.argv.length > 3) {
  var infile = process.argv[2];
  var outfile = process.argv[3];
  var mpx = MP4Box.createFile();
  var arrayBuffer = new Uint8Array(fs.readFileSync(infile)).buffer;


  arrayBuffer.fileStart = 0;
  mpx.appendBuffer(arrayBuffer);

  mpx.onError = function(e) {
    console.log('mp4box failed to parse data.');
  };

  for (const track of mpx.moov.traks) {
    var codec = track.mdia.minf.stbl.stsd.entries[0].getCodec();
    console.log(codec);
    if (codec === 'gpmd') {
      console.log('Found gpmd entry');
      var payloads = track.samples.map((x) => {
        return {
          sz: x.size,
          track_id: x.track_id,
          offset: x.offset,
          timescale: x.timescale,
          duration: x.duration,
          cts: x.cts, 
          dts: x.dts,
        };
      });
      var data = {payloads: payloads, info: {filename: infile}};
      // console.log(samples[1]);
      fs.writeFile(outfile, JSON.stringify(data), () => {});
    }
  }
  mpx.flush();
  mpx.start();


  console.log(mpx.print(MP4Box.Log));
} else {
  console.log('usage: node info.js <file> <outfile>');
}
