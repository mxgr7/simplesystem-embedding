// Dump a small representative slice of prod for use as a fixture.
//
// Pulls SAMPLE_SIZE uniform-random offers from `prod.offers`, then for
// each offer fetches the matching rows from `prod.pricings`,
// `prod.coreArticleMarkers`, and `prod.customerArticleNumbers`
// (all joined on (articleNumber, vendorId)).
//
// Runs against the prod mongo via the jumphost; output is written to
// stdout as EJSON-compatible JSON. Wrap with `mongosh --quiet`.
//
//   ssh simplesystem-nextgen 'mongosh "<URI>" --quiet' < dump_mongo_sample.js > out.json
//
// Output shape:
//   { generated_at, sample_size,
//     records: [ { offer, pricings, markers, customerArticleNumbers } ] }

// Force the `prod` database — without this the script runs against
// whatever DB the URI defaults to (often `test`), which silently has no
// `offers` collection and blows up downstream with `offers.map is not a
// function`.
db = db.getSiblingDB("prod");

const SAMPLE_SIZE = 200;

const offers = db.offers
  .aggregate([{ $sample: { size: SAMPLE_SIZE } }], { allowDiskUse: true })
  .toArray();

const records = offers.map((off) => {
  const key = { articleNumber: off.articleNumber, vendorId: off.vendorId };
  const pricings = db.pricings.find(key).toArray();
  const markers = db.coreArticleMarkers.find(key).toArray();
  const customerArticleNumbers = db.customerArticleNumbers.find(key).toArray();
  return { offer: off, pricings, markers, customerArticleNumbers };
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
