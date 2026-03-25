--
-- PostgreSQL database dump
--

\restrict geVXUPU5tMQNZqmR0v94BYxHaCqJEfSsENwy36foKOF6cgtf4yw1G2sSOMY2Aul

-- Dumped from database version 18.3 (Debian 18.3-1.pgdg13+1)
-- Dumped by pg_dump version 18.3 (Debian 18.3-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: audit_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.audit_log (
    id bigint NOT NULL,
    run_id integer,
    action character varying(100) NOT NULL,
    detail text,
    status character varying(20) NOT NULL,
    ip_address character varying(45),
    user_agent character varying(500),
    duration_ms integer,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.audit_log OWNER TO postgres;

--
-- Name: COLUMN audit_log.action; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.action IS 'e.g. file_upload, pipeline_built, cpp_ranking_saved';


--
-- Name: COLUMN audit_log.detail; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.detail IS 'Free-text context';


--
-- Name: COLUMN audit_log.status; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.status IS 'success / failure / warning';


--
-- Name: COLUMN audit_log.ip_address; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.ip_address IS 'Client IP address (IPv4 or IPv6, max 45 chars)';


--
-- Name: COLUMN audit_log.user_agent; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.user_agent IS 'HTTP User-Agent header truncated to 500 chars';


--
-- Name: COLUMN audit_log.duration_ms; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.audit_log.duration_ms IS 'Request processing time in milliseconds';


--
-- Name: audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.audit_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.audit_log_id_seq OWNER TO postgres;

--
-- Name: audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.audit_log_id_seq OWNED BY public.audit_log.id;


--
-- Name: batch_runs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.batch_runs (
    id integer NOT NULL,
    run_name character varying(255),
    filename character varying(500),
    data_format character varying(20) NOT NULL,
    n_batches integer NOT NULL,
    n_cpps integer NOT NULL,
    n_time_steps integer,
    has_yield boolean NOT NULL,
    golden_pct_used double precision,
    cpp_vars json,
    pipeline_meta json,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.batch_runs OWNER TO postgres;

--
-- Name: COLUMN batch_runs.run_name; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.run_name IS 'User-provided label';


--
-- Name: COLUMN batch_runs.filename; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.filename IS 'Original uploaded filename';


--
-- Name: COLUMN batch_runs.data_format; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.data_format IS '''long'' or ''wide''';


--
-- Name: COLUMN batch_runs.n_time_steps; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.n_time_steps IS 'Avg time steps per batch';


--
-- Name: COLUMN batch_runs.golden_pct_used; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.golden_pct_used IS 'Yield percentile used for golden profile';


--
-- Name: COLUMN batch_runs.cpp_vars; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.cpp_vars IS 'List of CPP column names';


--
-- Name: COLUMN batch_runs.pipeline_meta; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batch_runs.pipeline_meta IS 'Imputation counts, outlier info, etc.';


--
-- Name: batch_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.batch_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.batch_runs_id_seq OWNER TO postgres;

--
-- Name: batch_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.batch_runs_id_seq OWNED BY public.batch_runs.id;


--
-- Name: batches; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.batches (
    id integer NOT NULL,
    run_id integer NOT NULL,
    batch_id character varying(255) NOT NULL,
    yield_value double precision,
    quality character varying(20),
    is_golden boolean NOT NULL,
    duration integer,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.batches OWNER TO postgres;

--
-- Name: COLUMN batches.batch_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batches.batch_id IS 'Business batch identifier';


--
-- Name: COLUMN batches.quality; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batches.quality IS 'Golden / Good / Poor / Unknown';


--
-- Name: COLUMN batches.duration; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.batches.duration IS 'Number of time steps';


--
-- Name: batches_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.batches_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.batches_id_seq OWNER TO postgres;

--
-- Name: batches_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.batches_id_seq OWNED BY public.batches.id;


--
-- Name: cpp_rankings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cpp_rankings (
    id integer NOT NULL,
    run_id integer NOT NULL,
    variable character varying(255) NOT NULL,
    rank integer NOT NULL,
    corr_score double precision,
    tree_score double precision,
    pca_score double precision,
    lasso_score double precision,
    avg_score double precision,
    method_agreement integer,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cpp_rankings OWNER TO postgres;

--
-- Name: COLUMN cpp_rankings.variable; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.cpp_rankings.variable IS 'CPP column name';


--
-- Name: COLUMN cpp_rankings.avg_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.cpp_rankings.avg_score IS 'Weighted combined score';


--
-- Name: COLUMN cpp_rankings.method_agreement; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.cpp_rankings.method_agreement IS '0-4: how many methods agree';


--
-- Name: cpp_rankings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cpp_rankings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.cpp_rankings_id_seq OWNER TO postgres;

--
-- Name: cpp_rankings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cpp_rankings_id_seq OWNED BY public.cpp_rankings.id;


--
-- Name: golden_profile_meta; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.golden_profile_meta (
    id integer NOT NULL,
    run_id integer NOT NULL,
    n_golden integer NOT NULL,
    yield_threshold double precision,
    t_crit double precision,
    golden_batch_ids json,
    cpp_stats json,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.golden_profile_meta OWNER TO postgres;

--
-- Name: COLUMN golden_profile_meta.n_golden; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.golden_profile_meta.n_golden IS 'Number of golden batches used';


--
-- Name: COLUMN golden_profile_meta.yield_threshold; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.golden_profile_meta.yield_threshold IS 'Yield cutoff for golden batches';


--
-- Name: COLUMN golden_profile_meta.t_crit; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.golden_profile_meta.t_crit IS 't-distribution critical value used';


--
-- Name: COLUMN golden_profile_meta.golden_batch_ids; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.golden_profile_meta.golden_batch_ids IS 'List of batch_ids in golden set';


--
-- Name: COLUMN golden_profile_meta.cpp_stats; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.golden_profile_meta.cpp_stats IS 'Per-CPP mean/std/upper/lower arrays (serialised)';


--
-- Name: golden_profile_meta_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.golden_profile_meta_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.golden_profile_meta_id_seq OWNER TO postgres;

--
-- Name: golden_profile_meta_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.golden_profile_meta_id_seq OWNED BY public.golden_profile_meta.id;


--
-- Name: audit_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log ALTER COLUMN id SET DEFAULT nextval('public.audit_log_id_seq'::regclass);


--
-- Name: batch_runs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batch_runs ALTER COLUMN id SET DEFAULT nextval('public.batch_runs_id_seq'::regclass);


--
-- Name: batches id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batches ALTER COLUMN id SET DEFAULT nextval('public.batches_id_seq'::regclass);


--
-- Name: cpp_rankings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cpp_rankings ALTER COLUMN id SET DEFAULT nextval('public.cpp_rankings_id_seq'::regclass);


--
-- Name: golden_profile_meta id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.golden_profile_meta ALTER COLUMN id SET DEFAULT nextval('public.golden_profile_meta_id_seq'::regclass);


--
-- Data for Name: audit_log; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.audit_log (id, run_id, action, detail, status, ip_address, user_agent, duration_ms, created_at) FROM stdin;
\.


--
-- Data for Name: batch_runs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.batch_runs (id, run_name, filename, data_format, n_batches, n_cpps, n_time_steps, has_yield, golden_pct_used, cpp_vars, pipeline_meta, created_at) FROM stdin;
\.


--
-- Data for Name: batches; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.batches (id, run_id, batch_id, yield_value, quality, is_golden, duration, created_at) FROM stdin;
\.


--
-- Data for Name: cpp_rankings; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cpp_rankings (id, run_id, variable, rank, corr_score, tree_score, pca_score, lasso_score, avg_score, method_agreement, created_at) FROM stdin;
\.


--
-- Data for Name: golden_profile_meta; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.golden_profile_meta (id, run_id, n_golden, yield_threshold, t_crit, golden_batch_ids, cpp_stats, created_at) FROM stdin;
\.


--
-- Name: audit_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.audit_log_id_seq', 1, false);


--
-- Name: batch_runs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.batch_runs_id_seq', 1, false);


--
-- Name: batches_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.batches_id_seq', 1, false);


--
-- Name: cpp_rankings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cpp_rankings_id_seq', 1, false);


--
-- Name: golden_profile_meta_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.golden_profile_meta_id_seq', 1, false);


--
-- Name: audit_log audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);


--
-- Name: batch_runs batch_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batch_runs
    ADD CONSTRAINT batch_runs_pkey PRIMARY KEY (id);


--
-- Name: batches batches_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batches
    ADD CONSTRAINT batches_pkey PRIMARY KEY (id);


--
-- Name: cpp_rankings cpp_rankings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cpp_rankings
    ADD CONSTRAINT cpp_rankings_pkey PRIMARY KEY (id);


--
-- Name: golden_profile_meta golden_profile_meta_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.golden_profile_meta
    ADD CONSTRAINT golden_profile_meta_pkey PRIMARY KEY (id);


--
-- Name: golden_profile_meta golden_profile_meta_run_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.golden_profile_meta
    ADD CONSTRAINT golden_profile_meta_run_id_key UNIQUE (run_id);


--
-- Name: batches uq_run_batch; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batches
    ADD CONSTRAINT uq_run_batch UNIQUE (run_id, batch_id);


--
-- Name: cpp_rankings uq_run_cpp; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cpp_rankings
    ADD CONSTRAINT uq_run_cpp UNIQUE (run_id, variable);


--
-- Name: audit_log audit_log_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.batch_runs(id) ON DELETE SET NULL;


--
-- Name: batches batches_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batches
    ADD CONSTRAINT batches_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.batch_runs(id) ON DELETE CASCADE;


--
-- Name: cpp_rankings cpp_rankings_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cpp_rankings
    ADD CONSTRAINT cpp_rankings_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.batch_runs(id) ON DELETE CASCADE;


--
-- Name: golden_profile_meta golden_profile_meta_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.golden_profile_meta
    ADD CONSTRAINT golden_profile_meta_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.batch_runs(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict geVXUPU5tMQNZqmR0v94BYxHaCqJEfSsENwy36foKOF6cgtf4yw1G2sSOMY2Aul

