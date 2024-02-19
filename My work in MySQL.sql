--								SQL step-project

# Queries

/* 1. Show the average salary of employees for each year (average salary
among those who worked during the reporting period - statistics from the beginning to 2005). */

USE employees;

SELECT YEAR(s.from_date) AS report_year,
       ROUND(AVG(s.salary), 2) AS "average salary"
FROM employees.salaries AS s
GROUP BY 1
HAVING report_year BETWEEN MIN(report_year) AND 2005
ORDER BY report_year;

/* 2. Show the average employee salary for each department. Note: accept in
Calculation only of current departments and current wages. */

SELECT de.dept_no, d.dept_name, ROUND(AVG(s.salary), 2) AS "average salary"
FROM employees.dept_emp AS de
INNER JOIN employees.departments AS d 
   ON (de.dept_no = d.dept_no)
INNER JOIN employees.salaries AS s 	
   ON (de.emp_no = s.emp_no) AND (CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
WHERE CURRENT_DATE() BETWEEN de.from_date AND de.to_date
GROUP BY 1, 2
ORDER BY de.dept_no;

/* 3. Show the average employee salary for each department for each year.
Note: For the average salary of department X in year Y, we need to take the average
the value of all salaries in year Y of employees who were in department X in year Y. */

SELECT de.dept_no, dn.dept_name,       
	   IF(YEAR(CURRENT_DATE()) BETWEEN YEAR(s.from_date) AND YEAR(s.to_date), "current year", YEAR(s.to_date)) AS report_year,       
	   ROUND(AVG(s.salary), 2) AS 'average salary'
FROM employees.dept_emp AS de
INNER JOIN employees.salaries AS s 
	ON (de.emp_no = s.emp_no)
INNER JOIN employees.departments AS dn 
	ON (de.dept_no = dn.dept_no)
GROUP BY 1, 3
ORDER BY de.dept_no;

/* 4. Show for each year the largest department (by number of employees) in that
year and his average salary. */

WITH sub AS (    
			 SELECT de.dept_no, d.dept_name,    
  		            EXTRACT(YEAR FROM s.to_date) AS report_year, 
		            COUNT(de.emp_no) AS empl_count,   
		            ROUND(AVG(s.salary), 2) AS average_salary    	         
			 FROM employees.dept_emp AS de    
   	         INNER JOIN employees.departments AS d 
				ON (d.dept_no = de.dept_no)   
	         INNER JOIN employees.salaries AS s 
				ON (de.emp_no = s.emp_no)    
			 GROUP BY 1, 3
), max_count AS (    
	             SELECT report_year, MAX(empl_count) AS max_empl_count    
	             FROM sub    
	             GROUP BY 1
)
SELECT sub.dept_no, sub.dept_name,          
       CASE 
			WHEN YEAR(CURRENT_DATE()) BETWEEN sub.report_year AND sub.report_year
			THEN "current year"
			ELSE sub.report_year
	   END AS report_year, 
       sub.empl_count, sub.average_salary
FROM sub
INNER JOIN max_count    
	ON (sub.report_year = max_count.report_year)    
		AND (sub.empl_count = max_count.max_empl_count)
ORDER BY report_year;

# 5. Show details of the longest-serving manager responsibilities at the moment. */

SELECT e.emp_no, CONCAT(e.first_name, ' ', e.last_name) AS full_name, e.gender, e.birth_date,
       e.hire_date, dm.dept_no, d.dept_name, t.title, t.from_date, CURRENT_DATE() AS to_date,
       s.salary,  CONCAT(FORMAT(TIMESTAMPDIFF(DAY, t.from_date, CURRENT_DATE()) / 365, 2), ' years') AS task_completion
FROM employees.employees AS e
INNER JOIN employees.dept_manager AS dm          	
	ON (e.emp_no = dm.emp_no) AND (CURRENT_DATE() BETWEEN dm.from_date AND dm.to_date)
INNER JOIN employees.departments AS d     	
	ON (dm.dept_no = d.dept_no)
INNER JOIN employees.titles AS t     	
	ON (e.emp_no = t.emp_no) AND (CURRENT_DATE() BETWEEN t.from_date AND t.to_date)
INNER JOIN employees.salaries AS s       	
	ON (e.emp_no = s.emp_no) AND (CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
ORDER BY task_completion DESC
LIMIT 1;

/* 6. Show the top 10 current employees of the company with the greatest difference between their
salary and the current average salary in their department. */

WITH DeptSalary AS (	
	SELECT de.dept_no,  AVG(s.salary) AS avg_salary    
	FROM employees.salaries AS s    
	INNER JOIN employees.dept_emp de 
	   ON (s.emp_no = de.emp_no)   
		AND (CURRENT_DATE() BETWEEN de.from_date AND de.to_date) 
		AND (CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
	GROUP BY 1
)
SELECT s.emp_no, CONCAT(e.first_name, ' ', e.last_name) AS full_name, ds.dept_no, 	
		 ROUND((s.salary - ds.avg_salary), 2) AS difference
FROM employees.salaries AS s
INNER JOIN employees.dept_emp AS de 
	ON (s.emp_no = de.emp_no)  AND (CURRENT_DATE() BETWEEN de.from_date AND de.to_date)     
							   AND (CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
INNER JOIN DeptSalary AS ds 
	ON (ds.dept_no = de.dept_no)
INNER JOIN employees.departments AS d 
	ON (d.dept_no = de.dept_no)
INNER JOIN employees.employees AS e  
	ON (s.emp_no = e.emp_no)
ORDER BY difference DESC
LIMIT 10;

/* 7. Due to the crisis, one department is allocated funds for timely payment of salaries only 500 thousand dollars. 
The board decided that low-paid employees will be the first to receive their salary. 
Show a list of all employees who will be receive your salary on time (please note that we must pay salary for
one month, but we store annual amounts in the database). */

WITH SalaryData AS (    
		            SELECT s.emp_no, s.salary / 12 AS salary_month, de.dept_no,           
						   SUM(s.salary / 12) OVER (PARTITION BY de.dept_no ORDER BY (s.salary / 12) 
													ROWS  BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS sum_increasing    
		            FROM employees.salaries AS s          
		            INNER JOIN employees.dept_emp AS de             
	             	     ON (s.emp_no = de.emp_no) AND (CURRENT_DATE() BETWEEN de.from_date AND de.to_date)												AND CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
) 
SELECT sd.emp_no, CONCAT(e.first_name, ' ', e.last_name) AS full_name, sd.dept_no, d.dept_name,
       ROUND(sd.salary_month, 2) AS salary_month, ROUND(sd.sum_increasing, 2) AS sum_increasing
FROM SalaryData AS sd
INNER JOIN employees.employees AS e         
	ON (sd.emp_no = e.emp_no)
INNER JOIN employees.departments AS d         
	ON (sd.dept_no = d.dept_no)
WHERE sd.sum_increasing <= 500000
ORDER BY sd.dept_no, salary_month;

--							Database design:

/* 1. Develop a database for course management. Database contains
the following entities:
a. students: student_no, teacher_no, course_no, student_name, email, birth_date.
b. teachers: teacher_no, teacher_name, phone_no
c. courses: course_no, course_name, start_date, end_date.
● Partition by year, the students table by the birth_date field using the range mechanism
● In the students table, make a primary key in a combination of two fields student_no and birth_date
● Create an index on the students.email field
● Create a unique index on the teachers.phone_no field */

DROP DATABASE IF EXISTS management;
CREATE DATABASE IF NOT EXISTS management;

USE management;

DROP TABLE IF EXISTS students;
CREATE TABLE IF NOT EXISTS students (
    student_no INT AUTO_INCREMENT,
    teacher_no INT NOT NULL,
    course_no INT NOT NULL,
    student_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    birth_date DATE NOT NULL,
    PRIMARY KEY (student_no, birth_date)
)
PARTITION BY RANGE(YEAR(birth_date)) (
    PARTITION p_1990 VALUES LESS THAN (1991),
    PARTITION p_2000 VALUES LESS THAN (2001),
    PARTITION p_2010 VALUES LESS THAN (2011),
    PARTITION p_future VALUES LESS THAN (MAXVALUE)
);

CREATE INDEX idx_email ON students (email);

DROP TABLE IF EXISTS teachers;
CREATE TABLE IF NOT EXISTS teachers (
  teacher_no INT AUTO_INCREMENT PRIMARY KEY,
  teacher_name VARCHAR(100) NOT NULL,
  phone_no VARCHAR(20) NOT NULL
);
ALTER TABLE teachers AUTO_INCREMENT = 1000;

DROP TABLE IF EXISTS courses;
CREATE TABLE IF NOT EXISTS courses (
  course_no INT AUTO_INCREMENT PRIMARY KEY,
  course_name VARCHAR(100) NOT NULL,
  start_date DATE DEFAULT (CURRENT_DATE()),
  end_date DATE NOT NULL
);
ALTER TABLE courses AUTO_INCREMENT = 2000;

# 2. At your discretion, add test data (7-10 rows) to our three tables.

START TRANSACTION;

INSERT INTO students (teacher_no, course_no, student_name, email, birth_date) 
VALUES
  (1000, 2001, 'John Smith', 'johnsmith@example.com', '1992-05-15'),
  (1000, 2001, 'Jane Doe', 'janedoe@example.com', '1994-09-20'),
  (1000, 2002, 'Michael Johnson', 'michaeljohnson@example.com', '1995-02-10'),
  (1001, 2003, 'Emily Williams', 'emilywilliams@example.com', '1993-11-05'),
  (1001, 2003, 'David Brown', 'davidbrown@example.com', '1996-07-25'),
  (1001, 2003, 'Olivia Taylor', 'oliviataylor@example.com', '1997-03-30'),
  (1002, 2004, 'Sophia Anderson', 'sophiaanderson@example.com', '1995-08-12'),
  (1002, 2004, 'James Wilson', 'jameswilson@example.com', '1992-12-01'),
  (1002, 2004, 'Emma Martinez', 'emmamartinez@example.com', '1994-04-18'),
  (1002, 2005, 'Daniel Lee', 'daniellee@example.com', '1996-06-08');

INSERT INTO teachers (teacher_name, phone_no) 
VALUES
  ('John Doe', '123-456-7890'),
  ('Mary Johnson', '987-654-3210'),
  ('Robert Smith', '555-123-4567');

ALTER TABLE teachers 
	ADD COLUMN phone_prefix VARCHAR(5);

UPDATE teachers 
	SET phone_prefix = LEFT(phone_no, 3);

CREATE UNIQUE INDEX idx_phone_prefix 
	ON teachers (phone_prefix);

INSERT INTO courses (course_name, end_date) 
VALUES
  ('Mathematics', '2023-07-30'),
  ('English Literature', '2023-08-15'),
  ('Science', '2023-07-31');

COMMIT;

/* 3. Display data for any year from the students table and record it in mind comment on the query execution plan, 
where it will be clear that the query will be executed according to specific section. */

SELECT s.student_no, s.teacher_no, s.course_no, s.student_name, s.email, s.birth_date
FROM management.students AS s
WHERE s.birth_date BETWEEN '1994-01-01' AND '1994-12-31';

EXPLAIN SELECT s.student_no, s.teacher_no, s.course_no, s.student_name, s.email, s.birth_date
FROM management.students AS s
WHERE s.birth_date BETWEEN '1994-01-01' AND '1994-12-31';

/* 4. Display the teacher’s data for any one phone number and record the plan query execution, where 
it will be clear that the query will be executed by index, and not using the ALL method. 
Next, make the index from the teachers.phone_no field invisible and fix the query execution plan, 
where the expected result is the ALL method. Eventually Leave the index in the visible status. */

SELECT t.teacher_name, t.phone_no
FROM management.teachers AS t
WHERE t.phone_prefix = '123';

EXPLAIN SELECT t.teacher_name, t.phone_no
FROM management.teachers AS t
WHERE t.phone_prefix = '123';

ALTER TABLE management.teachers 
	ALTER INDEX idx_phone_prefix INVISIBLE;

EXPLAIN SELECT t.teacher_name, t.phone_no
FROM management.teachers AS t
WHERE t.phone_prefix = '123';

ALTER TABLE management.teachers 
	ALTER INDEX idx_phone_prefix VISIBLE;

# 5. We will specially make 3 duplications in the students table (add 3 more identical rows).

ALTER TABLE management.students 
	MODIFY student_no INT;

ALTER TABLE management.students 
	DROP PRIMARY KEY;

START TRANSACTION;

INSERT INTO management.students (student_no, teacher_no, course_no, student_name, email, birth_date)
SELECT MAX(student_no) + 1, teacher_no, course_no, student_name, email, birth_date
FROM management.students
GROUP BY student_no, 2, 3, 4, 5, 6
LIMIT 3;

COMMIT;

-- Or
START TRANSACTION;

INSERT INTO management.students 
SELECT '10', '1002', '2005', 'Daniel Lee', 'daniellee@example.com', '1996-06-08'
FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3) AS multi;

COMMIT;

# 6. Write a query that displays lines with duplicates.

SELECT s.student_no, s.teacher_no, s.course_no, s.student_name, s.email, s.birth_date,
	   COUNT(s.student_no) AS count_dup
FROM management.students AS s
GROUP BY 1, 2, 3, 4, 5, 6
HAVING count_dup > 1;

SET @row_valuable = 0;

UPDATE management.students
	SET student_no = (@row_valuable := @row_valuable + 1);

ALTER TABLE management.students 
	ADD PRIMARY KEY (student_no, birth_date);
    
ALTER TABLE management.students 
	MODIFY student_no INT AUTO_INCREMENT;
