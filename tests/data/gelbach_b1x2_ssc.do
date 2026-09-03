version 16.1
clear all
set more off

* Run from the repository root with b1x2 4.1.0 installed (ssc install b1x2).
* Reference: Jonah B. Gelbach, https://ideas.repec.org/c/boc/bocode/s457814.html
* Temporary copies leave the installed ado unchanged. Both expose the existing
* full-model coefficient contribution to Covdelta. The "none" copy additionally
* sets every _robust call to minus(0), disabling both K and cluster corrections:
* https://www.stata.com/manuals/p_robust.pdf (Methods and formulas).
* This generator covers OLS HC1 and one-way CRV1, not IID or IV inference.

findfile b1x2.ado
local source `"`r(fn)'"'
tempfile default_ado none_ado results_file

mata:
void gelbach_reference_ado(string scalar source, string scalar target,
                          real scalar no_ssc)
{
    real scalar input, output, robust_calls, covariance_exports, assignments
    string scalar line, clean, bt, qt, beta, delta, fullcov, v2upart
    input = fopen(source, "r")
    output = fopen(target, "w")
    line = fget(input)
    assert(strpos(line, "*!Version 4.1.0 20Jan10") == 1)
    bt = char(96)
    qt = char(39)
    beta = bt + "reference_beta_vcov" + qt
    delta = bt + "Covdelta" + qt
    fullcov = bt + "fullcov" + qt
    v2upart = bt + "v2upart" + qt
    robust_calls = covariance_exports = assignments = 0
    while (line != J(0, 0, "")) {
        clean = strtrim(subinstr(line, char(9), " "))
        if (strpos(clean, "_robust ") == 1) {
            robust_calls++
            if (no_ssc) {
                line = regexr(line, "minus\([^)]*\)", "") + " minus(0)"
            }
        }
        fput(output, line)
        // Mirror Covdelta's group-to-variable ordering and total-effect sum.
        if (strpos(clean, "mat " + delta) == 1) {
            if (assignments == 0) fput(output, "tempname reference_beta_vcov")
            assignments++
            fput(output, subinstr(subinstr(line, delta, beta), fullcov, v2upart))
        }
        if (regexm(clean, "eret mat +Covdelta")) {
            covariance_exports++
            fput(output, "ereturn matrix Vdelta_beta " + beta)
        }
        line = fget(input)
    }
    fclose(input)
    fclose(output)
    printf("b1x2 source checks: %g _robust calls, %g covariance export\n", robust_calls, covariance_exports)
    assert(robust_calls == 6)
    assert(covariance_exports == 1)
    assert(assignments == 4)
}
end

mata: gelbach_reference_ado(st_local("source"), st_local("default_ado"), 0)
mata: gelbach_reference_ado(st_local("source"), st_local("none_ado"), 1)

tempname results
postfile `results' str10 weights_type str6 vcov str7 ssc ///
    str20 effect_i str20 effect_j ///
    double coefficient double covariance double beta_covariance ///
    using "`results_file'", replace

foreach correction in default none {
    * Isolated batch session: discard definitions before loading the next copy.
    program drop _all
    quietly run "``correction'_ado'"
    foreach weights_type in unweighted aweights fweights {
        use "tests/data/gelbach.dta", clear
        gen double aw = 0.75 + mod(_n, 7) / 4
        gen long fw = 1 + mod(_n, 3)
        local weight ""
        if "`weights_type'" == "aweights" local weight "[aweight=aw]"
        if "`weights_type'" == "fweights" local weight "[fweight=fw]"

        foreach vcov in hetero CRV1 {
            local vce "robust"
            if "`vcov'" == "CRV1" local vce "cluster(cluster)"
            quietly b1x2 y `weight', x1all(x1) x2all(x21 x22 x23) ///
                x2delta(g1 = x21 x22 : g2 = x23) x1only(x1) `vce'

            matrix b_base = e(b1base)
            matrix b_full = e(b1full)
            matrix V_base = e(V1base)
            matrix V_full = e(V1full)
            matrix delta = e(Delta)
            matrix V_delta = e(Covdelta)
            matrix V_beta = e(Vdelta_beta)
            assert colsof(delta) == 3
            assert rowsof(V_delta) == 3
            assert rowsof(V_beta) == 3

            post `results' ("`weights_type'") ("`vcov'") ("`correction'") ///
                ("direct_effect") ("direct_effect") (b_base[1, 1]) (V_base[1, 1]) (.)
            post `results' ("`weights_type'") ("`vcov'") ("`correction'") ///
                ("full_effect") ("full_effect") (b_full[1, 1]) (V_full[1, 1]) (.)
            local effects "g1 g2 explained_effect"
            forvalues i = 1/3 {
                local effect_i : word `i' of `effects'
                forvalues j = 1/3 {
                    local effect_j : word `j' of `effects'
                    post `results' ("`weights_type'") ("`vcov'") ("`correction'") ///
                        ("`effect_i'") ("`effect_j'") (delta[1, `i']) ///
                        (V_delta[`i', `j']) (V_beta[`i', `j'])
                }
            }
        }
    }
}

postclose `results'
use "`results_file'", clear
sort weights_type vcov ssc effect_i effect_j
format coefficient covariance beta_covariance %24.17g
export delimited using "tests/data/gelbach_b1x2_ssc.csv", replace datafmt
assert _N == 132
display "Gelbach SSC reference complete: " _N " rows"
