import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ i(X1) + X2", data=data)
fit.iplot(joint=True)
