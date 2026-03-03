-- public.telco_raw definição

-- Drop table

-- DROP TABLE public.telco_raw;

CREATE TABLE public.telco_raw (
	customerid text NULL,
	gender text NULL,
	seniorcitizen int4 NULL,
	partner text NULL,
	dependents text NULL,
	tenure int4 NULL,
	phoneservice text NULL,
	multiplelines text NULL,
	internetservice text NULL,
	onlinesecurity text NULL,
	onlinebackup text NULL,
	deviceprotection text NULL,
	techsupport text NULL,
	streamingtv text NULL,
	streamingmovies text NULL,
	contract text NULL,
	paperlessbilling text NULL,
	paymentmethod text NULL,
	monthlycharges float8 NULL,
	totalcharges text NULL,
	churn text NULL
);