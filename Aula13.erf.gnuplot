set table "Aula13.erf.table"; set format "%.5f"
set format "%.7e";; set samples 1000; set dummy x; plot [x=-15:40] erf(0.22*x) + 0.17*sin(5.5*x);
