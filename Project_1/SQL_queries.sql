SET PASSWORD FOR 'root'@'localhost' = 'root';
use tennis_data;
select * from competitor;
select * from complexes;
select * from doubles_data;

-- competitor table
-- 1 in competitior table query
SELECT comp_id, comp_name, category_name FROM competitor;
-- 2 count number of competitions in each category
SELECT  category_name,COUNT(comp_name) as number FROM competitor group by category_name;
-- 3 
select comp_name,`type` from competitor where type ='doubles';
-- 4
select category_name,comp_name from competitor where category_name='ITF Men';
-- 5
select parent_id,comp_id from competitor order by parent_id desc;
-- 6
select `type`,count(comp_id),count(comp_name) from competitor group by `type`;
-- 7 
select comp_name from competitor where parent_id = 'NA';

-- complex table
-- 1
select complex_name, venue_name from complexes;

-- 2
select complex_name, COUNT(venue_name) as number from complexes group by complex_name;

-- 3
select country_name,venue_name from complexes where country_name='Chile';

-- 4
select venue_name, timezone from complexes order by timezone;

-- 5
select complex_name, count(venue_name) from complexes group by complex_id, complex_name having COUNT(venue_name) > 1;

-- 6
select GROUP_CONCAT(venue_name), country_name from complexes group by country_name;

-- 7
select complex_name, GROUP_CONCAT(venue_name) from complexes group by complex_name;

-- doubles data
-- 1
select competitor_name, ranks, points from doubles_data order by points desc;

-- 2
select competitor_name, ranks from doubles_data where ranks <= 5 order by ranks asc;
-- rank(for getting only top 5)
select competitor_name, DENSE_RANK() OVER(order by ranks ASC) AS doubles_rank from doubles_data where ranks <= 5;

-- 3
select competitor_name as competitor_with_stable_rank from doubles_data where movement = 0;

-- 4
select country, sum(points) as total_points from doubles_data where country = 'Croatia';
-- for all countires
select country, sum(points) as total_points from doubles_data group by country;

-- 5
select country, count(competitor_id) as No_of_competitor from doubles_data group by country;

-- 6
select competitor_name, points from doubles_data order by points desc limit 7;
select competitior_name, competitions_played, points from doubles_data order by points desc;