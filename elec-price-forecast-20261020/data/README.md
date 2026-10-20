# Data

## Source

`data/raw/de_lu_price_raw.json` is downloaded from the open
[energy-charts.info](https://api.energy-charts.info/) API:

```
GET https://api.energy-charts.info/price?bzn=DE-LU&start=2024-01-01&end=2026-10-18
```

- Market: Germany/Luxembourg (DE-LU) day-ahead auction price
- Unit: EUR/MWh
- Underlying source: Bundesnetzagentur | SMARD.de
- Licence: CC BY 4.0
- No API key required

Refresh with:

```bash
python scripts/fetch_data.py --start 2024-01-01 --end 2026-10-18
```

## Notes on the raw file

The feed mixes hourly and 15-minute observations and contains DST transitions.
`elecprice.data.parse_energy_charts_json` normalises everything to a complete
hourly UTC grid (mean aggregation, short gaps interpolated up to 3h) and adds
local-time calendar columns for `Europe/Berlin`.

## Plugging in another market

Any CSV with two columns works — `ts_utc` (ISO timestamp, UTC) and `price`
(EUR/MWh or your local currency):

```python
from elecprice.data import load_prices

df = load_prices("data/raw/my_market.csv")
df = df[df["price"].notna()]
```

For Chinese provincial spot markets (山西 / 广东 / 山东 / 蒙西), export the
day-ahead clearing price at 15-minute or hourly granularity into the same
schema and the whole pipeline runs unchanged.
