-- Regional Surveillance Summary Query
-- Joins cleaned COVID-19 surveillance records with the WHO Region reference lookup table
-- Aggregates total recorded cases, total recorded deaths, and calculates Case Fatality Ratio (CFR %)

SELECT 
    r.region_code AS who_region_code,
    r.region_name AS who_region_name,
    COUNT(c.Country) AS total_weekly_reports,
    COUNT(DISTINCT c.Country) AS active_countries_monitored,
    CAST(SUM(c.New_cases) AS BIGINT) AS total_confirmed_cases,
    CAST(SUM(c.New_deaths) AS BIGINT) AS total_confirmed_deaths,
    ROUND((SUM(c.New_deaths) * 100.0) / NULLIF(SUM(c.New_cases), 0), 2) AS case_fatality_ratio_percent
FROM cleaned_covid_data c
INNER JOIN who_regions r 
    ON c.WHO_region = r.region_code
GROUP BY 
    r.region_code, 
    r.region_name
ORDER BY 
    total_confirmed_cases DESC;
