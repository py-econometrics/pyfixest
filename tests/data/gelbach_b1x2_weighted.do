version 18.0
clear all
set more off

capture log close
log using "tests/data/gelbach_b1x2_weighted.txt", text replace

which b1x2
use "tests/data/gelbach.dta", clear

gen double aw = 0.75 + mod(_n, 7) / 4
gen long fw = 1 + mod(_n, 3)

tempfile results_file
tempname results
postfile `results' str8 weights_type str32 effect double coefficient using "`results_file'", replace

foreach weights_type in aweights fweights {
    if "`weights_type'" == "aweights" {
        local stata_weight "aweight"
        local weight_variable "aw"
    }
    else {
        local stata_weight "fweight"
        local weight_variable "fw"
    }

    b1x2 y [`stata_weight'=`weight_variable'], x1all(x1) x2all(x21 x22 x23) ///
        x2delta(g1 = x21 x22 : g2 = x23) x1only(x1) robust

    matrix b1base = e(b1base)
    matrix b1full = e(b1full)
    matrix delta = e(Delta)

    scalar direct_effect = b1base[1, 1]
    scalar full_effect = b1full[1, 1]
    scalar g1 = delta[1, 1]
    scalar g2 = delta[1, 2]
    scalar explained_effect = delta[1, 3]
    scalar unexplained_effect = direct_effect - explained_effect

    assert reldif(full_effect, unexplained_effect) < 1e-10

    post `results' ("`weights_type'") ("direct_effect") (direct_effect)
    post `results' ("`weights_type'") ("full_effect") (full_effect)
    post `results' ("`weights_type'") ("explained_effect") (explained_effect)
    post `results' ("`weights_type'") ("unexplained_effect") (unexplained_effect)
    post `results' ("`weights_type'") ("g1") (g1)
    post `results' ("`weights_type'") ("g2") (g2)
}

postclose `results'
use "`results_file'", clear
sort weights_type effect
format coefficient %24.17g
export delimited using "tests/data/gelbach_b1x2_weighted.csv", replace datafmt
list, noobs abbreviate(32)

log close
