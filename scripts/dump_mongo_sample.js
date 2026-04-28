// Dump a small representative slice of prod for use as a fixture.
//
// Pulls SAMPLE_SIZE uniform-random offers from `prod.offers`, then for
// each offer fetches the matching rows from `prod.pricings` and
// `prod.coreArticleMarkers` (joined on (articleNumber, vendorId)).
//
// Runs against the prod mongo via the jumphost; output is written to
// stdout as EJSON-compatible JSON. Wrap with `mongosh --quiet`.
//
//   ssh simplesystem-nextgen 'mongosh "<URI>" --quiet' < dump_mongo_sample.js > out.json
//
// Output shape:
//   { generated_at, sample_size, records: [ { offer, pricings, markers } ] }

const SAMPLE_SIZE = 200;

const offers = db.offers
  .aggregate([{ $sample: { size: SAMPLE_SIZE } }], { allowDiskUse: true })
  .toArray();

const records = offers.map((off) => {
  const key = { articleNumber: off.articleNumber, vendorId: off.vendorId };
  const pricings = db.pricings.find(key).toArray();
  const markers = db.coreArticleMarkers.find(key).toArray();
  return { offer: off, pricings, markers };
});

print(
  EJSON.stringify(
    {
      generated_at: new Date().toISOString(),
      sample_size: records.length,
      records,
    },
    null,
    2,
  ),
);
