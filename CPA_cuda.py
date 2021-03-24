import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


def cpa_cuda_256(traces: np.ndarray,
                 estimated_power_consumption: np.ndarray
                 ) -> np.ndarray:
    # traces : shape=(num_of_traces, num_of_samples)
    # estimated_power_consumption : shape=(num_of_traces, 256)
    # return: shape=(256, num_of_samples)

    if traces.flags['C_CONTIGUOUS'] is False or estimated_power_consumption.flags['C_CONTIGUOUS'] is False:
        raise RuntimeError("'traces' or 'estimated_power_consumption' are not contiguously allocated in memory."
                           "Please use 'np.ascontiguousarray(...)' to make the array contiguous.")

    result_corr = np.empty(shape=(256, traces.shape[1]))
    kernel_code = """
    __global__ void calc_corr(double *result, double *trace, double *exp_pc)
    {{
        // threadIdx.x : 0 ~ 256
        // blockIdx.x  : 0 ~ #POI
        // gridDim.x   : #POI

        int threshold = gridDim.x * {traceNum}, i = blockIdx.x, j = threadIdx.x;
        double sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, x, y;

        while( (i < threshold) && (j < threshold))
        {{
            x = trace[i];
            y = exp_pc[j];
            sumXY += x * y;
            sumX += x;
            sumY += y;
            sumX2 += x * x;
            sumY2 += y * y;
            i += gridDim.x;
            j += 256;
        }}

        x = ({traceNum} * sumXY - sumX * sumY);
        y = sqrt( ({traceNum} * sumX2 - sumX * sumX) * ({traceNum} * sumY2 - sumY * sumY) );
        result[gridDim.x * threadIdx.x + blockIdx.x] = x / y;
    }}
    """.format(traceNum=traces.shape[0])
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function("calc_corr")
    num_of_samples = traces.shape[1]

    calculate_cor(drv.Out(result_corr), drv.In(traces), drv.In(estimated_power_consumption),
                  block=(256, 1, 1), grid=(num_of_samples, 1, 1))

    return np.nan_to_num(result_corr, 0)
