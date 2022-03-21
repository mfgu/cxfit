from pylab import *
from cxfit import *

e2 = 50
m = 0
z,r1 = lznm1e(e2, 'D', md=m)
z,r2 = lznm1e(e2, 'D_2', md=m)
z,r3 = lznm1e(e2, 'D_3', md=m)
z,r4 = lznm1e(e2, 'D_4', md=m)
z,r5 = lznm1e(e2, 'D_5', md=m)

clf()
plot(z, r1, marker='o', label='n=1')
plot(z, r2, marker='o', label='n=2')
plot(z, r3, marker='o', label='n=3')
plot(z, r4, marker='o', label='n=4')
plot(z, r5, marker='o', label='n=5')
xlabel('Z')
ylabel('max n')
legend()
savefig('nm500.pdf')

