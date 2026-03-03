-- public.vw_telco_features fonte

CREATE OR REPLACE VIEW public.vw_telco_features
AS WITH base_calculations AS (
         SELECT vw_telco_clean.customerid,
            vw_telco_clean.gender,
            vw_telco_clean.seniorcitizen,
            vw_telco_clean.partner,
            vw_telco_clean.dependents,
            vw_telco_clean.tenure,
            vw_telco_clean.phoneservice,
            vw_telco_clean.multiplelines,
            vw_telco_clean.internetservice,
            vw_telco_clean.onlinesecurity,
            vw_telco_clean.onlinebackup,
            vw_telco_clean.deviceprotection,
            vw_telco_clean.techsupport,
            vw_telco_clean.streamingtv,
            vw_telco_clean.streamingmovies,
            vw_telco_clean.contract,
            vw_telco_clean.paperlessbilling,
            vw_telco_clean.paymentmethod,
            vw_telco_clean.monthlycharges,
            vw_telco_clean.totalcharges,
            vw_telco_clean.churn,
                CASE
                    WHEN vw_telco_clean.onlinesecurity = 'Yes'::text THEN 1
                    ELSE 0
                END +
                CASE
                    WHEN vw_telco_clean.onlinebackup = 'Yes'::text THEN 1
                    ELSE 0
                END +
                CASE
                    WHEN vw_telco_clean.deviceprotection = 'Yes'::text THEN 1
                    ELSE 0
                END +
                CASE
                    WHEN vw_telco_clean.techsupport = 'Yes'::text THEN 1
                    ELSE 0
                END +
                CASE
                    WHEN vw_telco_clean.streamingtv = 'Yes'::text THEN 1
                    ELSE 0
                END +
                CASE
                    WHEN vw_telco_clean.streamingmovies = 'Yes'::text THEN 1
                    ELSE 0
                END AS total_services
           FROM vw_telco_clean
        )
 SELECT customerid,
    gender,
    seniorcitizen,
    partner,
    dependents,
    tenure,
    phoneservice,
    multiplelines,
    internetservice,
    onlinesecurity,
    onlinebackup,
    deviceprotection,
    techsupport,
    streamingtv,
    streamingmovies,
    contract,
    paperlessbilling,
    paymentmethod,
    monthlycharges,
    totalcharges,
    churn,
    total_services,
        CASE
            WHEN paymentmethod ~~ '%automatic%'::text THEN 1
            ELSE 0
        END AS automatic_payment,
        CASE
            WHEN total_services > 0 THEN monthlycharges / total_services::double precision
            ELSE monthlycharges
        END AS avg_charge_per_service,
        CASE
            WHEN streamingtv = 'Yes'::text OR streamingmovies = 'Yes'::text THEN 1
            ELSE 0
        END AS streaming_user,
        CASE
            WHEN internetservice <> 'No'::text AND phoneservice = 'Yes'::text THEN 1
            ELSE 0
        END AS internet_and_phone,
        CASE
            WHEN tenure <= 6 THEN 'Newbie'::text
            WHEN tenure >= 7 AND tenure <= 24 THEN 'Stable'::text
            ELSE 'Loyal'::text
        END AS customer_lifecycle_stage
   FROM base_calculations;