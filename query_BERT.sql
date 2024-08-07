select Id as review_id, toDateTime(Time) as dt,
Score as rating, 
multiIf(Score = 5, 'positive', Score = 1, 'negative', 'neutral') AS sentiment,
Text as review
from simulator.flyingfood_reviews
order by review_id