-- Instamart Copilot - Part A SQL analysis (SQLite 3.25+).
-- Each query starts with a line "-- name: Qn"; src/part_a_analysis.py splits on these markers.
-- Unless stated otherwise only completed orders count as revenue or spend. Amounts are in INR.

-- name: Q1
-- WHERE filters rows before grouping; HAVING filters the groups afterwards.
-- Customers whose completed orders above INR 500 add up to more than INR 8000.
SELECT customer_id,
       COUNT(*)                     AS big_orders,
       ROUND(SUM(order_amount), 2)  AS total_big_orders_inr
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_big_orders_inr DESC;

-- name: Q2
-- Three-table join: completed revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;

-- name: Q3
-- Scalar subquery: products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- LEFT JOIN: customers who have no completed order (the status filter sits in the ON clause).
SELECT c.customer_id, c.city, c.signup_date
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
-- Two CTEs: per-customer completed spend, then the average of those totals.
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend_inr,
       ROUND(a.avg_spend, 2)   AS average_spend_inr
FROM customer_totals t
CROSS JOIN average_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- name: Q6
-- Ranking window functions within each city by total completed spend.
WITH customer_totals AS (
    SELECT c.city, o.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city,
       customer_id,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_totals
ORDER BY city, row_num;

-- name: Q7
-- Running total: daily completed revenue and its cumulative sum.
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day,
       ROUND(revenue, 2)                             AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)    AS cumulative_revenue_inr
FROM daily
ORDER BY day;
