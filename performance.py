original = {
    "bicg_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+------------------------------------+---+--------------+-----+------------------+
|             Hierarchy              |TC |      AC      | CPC |      Detail      |
+------------------------------------+---+--------------+-----+------------------+
|kernel_bicg (cnn-krnl.cpp:4)        |   |81467 (100.0%)|81467|-                 |
|    auto memory burst for 's'(read) |   |   14 (  0.0%)|   14|cache size=928B   |
|    loop i (cnn-krnl.cpp:11)        |116|  116 (  0.1%)|  116|pipeline II=1     |
|    auto memory burst for 'q'(read) |   |   15 (  0.0%)|   15|cache size=992B   |
|    auto memory burst for 'p'(read) |   |   14 (  0.0%)|   14|cache size=928B   |
|    auto memory burst for 'A'(read) |   | 3596 (  4.4%)| 3596|cache size=115072B|
|    auto memory burst for 'A'(read) |   | 3596 (  4.4%)| 3596|cache size=115072B|
|    auto memory burst for 'r'(read) |   |   15 (  0.0%)|   15|cache size=992B   |
|    loop i (cnn-krnl.cpp:20)        |124|74028 ( 90.9%)|74028|-                 |
|        loop j (cnn-krnl.cpp:24)    |116|73036 ( 89.7%)|  589|pipeline II=5     |
|    auto memory burst for 'q'(write)|   |   15 (  0.0%)|   15|cache size=992B   |
|    auto memory burst for 's'(write)|   |   14 (  0.0%)|   14|cache size=928B   |
+------------------------------------+---+--------------+-----+------------------+
""",
    "doitgen_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+----------------------------------------+--+---------------+------+-------------------+
|               Hierarchy                |TC|      AC       | CPC  |      Detail       |
+----------------------------------------+--+---------------+------+-------------------+
|kernel_doitgen (cnn-krnl.cpp:4)         |  |525156 (100.0%)|525156|-                  |
|    auto memory burst for 'sum'(read)   |  |     3 (  0.0%)|     3|cache size=240B    |
|    auto memory burst for 'A'(read)     |  |  7500 (  1.4%)|  7500|cache size=120000B |
|    auto memory burst for 'C4'(read)    |  |   450 (  0.1%)|   450|cache size=7200B   |
|    loop r (cnn-krnl.cpp:15)            |25|510001 ( 97.1%)|510001|pipeline II=34[1]  |
|        loop q (cnn-krnl.cpp:20)        |20|              -|     -|-                  |
|            loop p (cnn-krnl.cpp:25)    |30|              -|     -|-                  |
|                loop s (cnn-krnl.cpp:28)|30|              -|     -|-                  |
|            loop p (cnn-krnl.cpp:33)    |30|              -|     -|parallel factor=30x|
|    auto memory burst for 'A'(write)    |  |  7500 (  1.4%)|  7500|cache size=120000B |
|    auto memory burst for 'sum'(write)  |  |     3 (  0.0%)|     3|cache size=240B    |
+----------------------------------------+--+---------------+------+-------------------+
""",
    "atax_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+--------------------------------------+---+--------------+-----+------------------+
|              Hierarchy               |TC |      AC      | CPC |      Detail      |
+--------------------------------------+---+--------------+-----+------------------+
|kernel_atax (cnn-krnl.cpp:5)          |   |97221 (100.0%)|97221|-                 |
|    auto memory burst for 'y'(read)   |   |   15 (  0.0%)|   15|cache size=992B   |
|    loop i (cnn-krnl.cpp:10)          |124|  124 (  0.1%)|  124|pipeline II=1     |
|    auto memory burst for 'A'(read)   |   | 3596 (  3.7%)| 3596|cache size=115072B|
|    auto memory burst for 'x'(read)   |   |   15 (  0.0%)|   15|cache size=992B   |
|    auto memory burst for 'tmp'(read) |   |   14 (  0.0%)|   14|cache size=928B   |
|    auto memory burst for 'A'(read)   |   | 3596 (  3.7%)| 3596|cache size=115072B|
|    loop i (cnn-krnl.cpp:18)          |116|89784 ( 92.4%)|89784|-                 |
|        loop j (cnn-krnl.cpp:22)      |124|72848 ( 74.9%)|  628|pipeline II=5     |
|        loop j (cnn-krnl.cpp:27)      |124|15776 ( 16.2%)|  136|pipeline II=1     |
|    auto memory burst for 'tmp'(write)|   |   14 (  0.0%)|   14|cache size=928B   |
|    auto memory burst for 'y'(write)  |   |   15 (  0.0%)|   15|cache size=992B   |
+--------------------------------------+---+--------------+-----+------------------+
""",
    "gemver_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+------------------------------------+---+---------------+------+------------------+
|             Hierarchy              |TC |      AC       | CPC  |      Detail      |
+------------------------------------+---+---------------+------+------------------+
|kernel_gemver (cnn-krnl.cpp:5)      |   |167415 (100.0%)|167415|-                 |
|    auto memory burst for 'v1'(read)|   |    15 (  0.0%)|    15|cache size=960B   |
|    auto memory burst for 'x'(read) |   |    15 (  0.0%)|    15|cache size=960B   |
|    auto memory burst for 'v2'(read)|   |    15 (  0.0%)|    15|cache size=960B   |
|    auto memory burst for 'u2'(read)|   |    15 (  0.0%)|    15|cache size=960B   |
|    auto memory burst for 'A'(read) |   |  1800 (  1.1%)|  1800|cache size=115200B|
|    auto memory burst for 'u1'(read)|   |    15 (  0.0%)|    15|cache size=960B   |
|    loop i (cnn-krnl.cpp:17)        |120| 14417 (  8.6%)| 14417|pipeline II=1     |
|        loop j (cnn-krnl.cpp:20)    |120|              -|     -|flattened         |
|    auto memory burst for 'y'(read) |   |    15 (  0.0%)|    15|cache size=960B   |
|    loop i (cnn-krnl.cpp:30)        |120| 74520 ( 44.5%)| 74520|-                 |
|        loop j (cnn-krnl.cpp:33)    |120| 73680 ( 44.0%)|   614|pipeline II=5     |
|    loop i (cnn-krnl.cpp:39)        |120|   126 (  0.1%)|   126|pipeline II=1     |
|    auto memory burst for 'w'(read) |   |    15 (  0.0%)|    15|cache size=960B   |
|    loop i (cnn-krnl.cpp:48)        |120| 74520 ( 44.5%)| 74520|-                 |
|        loop j (cnn-krnl.cpp:51)    |120| 73680 ( 44.0%)|   614|pipeline II=5     |
|    auto memory burst for 'A'(write)|   |  1800 (  1.1%)|  1800|cache size=115200B|
|    auto memory burst for 'x'(write)|   |    15 (  0.0%)|    15|cache size=960B   |
|    auto memory burst for 'w'(write)|   |    15 (  0.0%)|    15|cache size=960B   |
+------------------------------------+---+---------------+------+------------------+
""",
    "syrk_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+------------------------------------+--+---------------+------+-----------------+
|             Hierarchy              |TC|      AC       | CPC  |     Detail      |
+------------------------------------+--+---------------+------+-----------------+
|kernel_syrk (cnn-krnl.cpp:4)        |  |396547 (100.0%)|396547|-                |
|    auto memory burst for 'A'(read) |  |  1200 (  0.3%)|  1200|cache size=38400B|
|    auto memory burst for 'C'(read) |  |   800 (  0.2%)|   800|cache size=51200B|
|    auto memory burst for 'A'(read) |  |  1200 (  0.3%)|  1200|cache size=38400B|
|    loop i (cnn-krnl.cpp:21)        |80|393120 ( 99.1%)|393120|-                |
|        loop j (cnn-krnl.cpp:24)    |80|  6960 (  1.8%)|    87|pipeline II=1    |
|        loop k (cnn-krnl.cpp:35)    |60|385520 ( 97.2%)|  4819|pipeline II=1    |
|            loop j (cnn-krnl.cpp:38)|80|              -|     -|flattened        |
|    auto memory burst for 'C'(write)|  |   800 (  0.2%)|   800|cache size=51200B|
+------------------------------------+--+---------------+------+-----------------+
""",
    "md_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+--------------------------------------------+---+--------------+-----+----------------+
|                 Hierarchy                  |TC |      AC      | CPC |     Detail     |
+--------------------------------------------+---+--------------+-----+----------------+
|md_kernel (cnn-krnl.cpp:4)                  |   |62545 (100.0%)|62545|-               |
|    auto memory burst for 'position_x'(read)|   |   32 (  0.1%)|   32|cache size=2048B|
|    auto memory burst for 'position_y'(read)|   |   32 (  0.1%)|   32|cache size=2048B|
|    auto memory burst for 'position_z'(read)|   |   32 (  0.1%)|   32|cache size=2048B|
|    loop loop_i (cnn-krnl.cpp:32)           |256|62436 ( 99.8%)|62436|pipelined       |
|        loop loop_j (cnn-krnl.cpp:40)       | 16|60160 ( 96.2%)|  235|pipeline II=5   |
+--------------------------------------------+---+--------------+-----+----------------+
""",
    "heat-3d_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+----------------------------------------+--+---------------+------+-----------------+
|               Hierarchy                |TC|      AC       | CPC  |     Detail      |
+----------------------------------------+--+---------------+------+-----------------+
|kernel_heat_3d (cnn-krnl.cpp:6)         |  |472868 (100.0%)|472868|-                |
|    auto memory burst for 'B'(read)     |  |   994 (  0.2%)|   994|cache size=63664B|
|    auto memory burst for 'A'(read)     |  |   994 (  0.2%)|   994|cache size=63664B|
|    loop t (cnn-krnl.cpp:19)            |40|469760 ( 99.3%)|469760|-                |
|        loop i (cnn-krnl.cpp:24)        |18|234720 ( 49.6%)|  5868|pipeline II=1    |
|            loop j (cnn-krnl.cpp:29)    |18|              -|     -|-                |
|                loop k (cnn-krnl.cpp:30)|18|              -|     -|-                |
|        loop i (cnn-krnl.cpp:39)        |18|234720 ( 49.6%)|  5868|pipeline II=1    |
|            loop j (cnn-krnl.cpp:44)    |18|              -|     -|-                |
|                loop k (cnn-krnl.cpp:45)|18|              -|     -|-                |
|    auto memory burst for 'B'(write)    |  |   894 (  0.2%)|   894|cache size=57264B|
|    auto memory burst for 'A'(write)    |  |   894 (  0.2%)|   894|cache size=57264B|
+----------------------------------------+--+---------------+------+-----------------+
""",
    "fdtd-2d_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+----------------------------------------+--+---------------+------+-----------------+
|               Hierarchy                |TC|      AC       | CPC  |     Detail      |
+----------------------------------------+--+---------------+------+-----------------+
|kernel_fdtd_2d (cnn-krnl.cpp:5)         |  |573320 (100.0%)|573320|-                |
|    auto memory burst for 'hz'(read)    |  |   600 (  0.1%)|   600|cache size=38400B|
|    auto memory burst for '_fict_'(read)|  |     5 (  0.0%)|     5|cache size=320B  |
|    auto memory burst for 'ex'(read)    |  |   600 (  0.1%)|   600|cache size=38400B|
|    auto memory burst for 'ey'(read)    |  |   600 (  0.1%)|   600|cache size=38400B|
|    loop t (cnn-krnl.cpp:17)            |40|571280 ( 99.6%)|571280|-                |
|        loop j (cnn-krnl.cpp:20)        |80|  3200 (  0.6%)|    80|pipeline II=1    |
|        loop i (cnn-krnl.cpp:29)        |59|189520 ( 33.1%)|  4738|pipeline II=1    |
|            loop j (cnn-krnl.cpp:32)    |80|              -|     -|flattened        |
|        loop i (cnn-krnl.cpp:42)        |60|190320 ( 33.2%)|  4758|pipeline II=1    |
|            loop j (cnn-krnl.cpp:45)    |79|              -|     -|flattened        |
|        loop i (cnn-krnl.cpp:55)        |59|187560 ( 32.7%)|  4689|pipeline II=1    |
|            loop j (cnn-krnl.cpp:58)    |79|              -|     -|flattened        |
|    auto memory burst for 'ex'(write)   |  |   599 (  0.1%)|   599|cache size=38392B|
|    auto memory burst for 'ey'(write)   |  |   600 (  0.1%)|   600|cache size=38400B|
|    auto memory burst for 'hz'(write)   |  |   589 (  0.1%)|   589|cache size=37752B|
+----------------------------------------+--+---------------+------+-----------------+
""",
    "stencil_stencil2d_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+-----------------------------------------------------+---+---------------+------+------------------+
|                      Hierarchy                      |TC |      AC       | CPC  |      Detail      |
+-----------------------------------------------------+---+---------------+------+------------------+
|stencil (cnn-krnl.cpp:4)                             |   |113737 (100.0%)|113737|-                 |
|    auto memory burst for 'orig'(read)               |   |   512 (  0.5%)|   512|cache size=32768B |
|    auto memory burst for 'filter'(read)             |   |     9 (  0.0%)|     9|cache size=36B    |
|    loop stencil_label1 (cnn-krnl.cpp:19)            |126|113148 ( 99.5%)|113148|-                 |
|        loop stencil_label2 (cnn-krnl.cpp:27)        | 62|112896 ( 99.3%)|   896|pipelined         |
|            loop stencil_label3 (cnn-krnl.cpp:32)    |  3| 46872 ( 41.2%)|     6|pipeline II=1     |
|                loop stencil_label4 (cnn-krnl.cpp:34)|  3|              -|     -|parallel factor=3x|
+-----------------------------------------------------+---+---------------+------+------------------+
""",
    "adi_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+------------------------------------+--+-----------------+--------+-------------------+
|             Hierarchy              |TC|       AC        |  CPC   |      Detail       |
+------------------------------------+--+-----------------+--------+-------------------+
|kernel_adi (cnn-krnl.cpp:5)         |  |14600697 (100.0%)|14600697|-                  |
|    auto memory burst for 'q'(read) |  |     869 (  0.0%)|     869|cache size=27832B  |
|    auto memory burst for 'p'(read) |  |     869 (  0.0%)|     869|cache size=27832B  |
|    auto memory burst for 'v'(read) |  |     899 (  0.0%)|     899|cache size=28784B  |
|    auto memory burst for 'u'(read) |  |     870 (  0.0%)|     870|cache size=27840B  |
|    loop t (cnn-krnl.cpp:43)        |40|14597560 (100.0%)|14597560|-                  |
|        loop i (cnn-krnl.cpp:51)    |58| 7298720 ( 50.0%)|  182468|-                  |
|            loop j (cnn-krnl.cpp:57)|58| 5813920 ( 39.8%)|    2506|pipeline II=43[1]  |
|            loop j (cnn-krnl.cpp:65)|58|                -|       -|parallel factor=58x|
|        loop i (cnn-krnl.cpp:78)    |58| 7298720 ( 50.0%)|  182468|-                  |
|            loop j (cnn-krnl.cpp:84)|58| 5813920 ( 39.8%)|    2506|pipeline II=43[2]  |
|            loop j (cnn-krnl.cpp:92)|58|                -|       -|parallel factor=58x|
|    auto memory burst for 'p'(write)|  |     869 (  0.0%)|     869|cache size=27832B  |
|    auto memory burst for 'u'(write)|  |     870 (  0.0%)|     870|cache size=27840B  |
|    auto memory burst for 'v'(write)|  |     899 (  0.0%)|     899|cache size=28784B  |
|    auto memory burst for 'q'(write)|  |     869 (  0.0%)|     869|cache size=27832B  |
+------------------------------------+--+-----------------+--------+-------------------+
""",
    "seidel-2d_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+------------------------------------+---+-----------------+--------+------------------+
|             Hierarchy              |TC |       AC        |  CPC   |      Detail      |
+------------------------------------+---+-----------------+--------+------------------+
|kernel_seidel_2d (cnn-krnl.cpp:6)   |   |30079636 (100.0%)|30079636|-                 |
|    auto memory burst for 'A'(read) |   |    1800 (  0.0%)|    1800|cache size=115200B|
|    loop t (cnn-krnl.cpp:18)        | 40|30075841 (100.0%)|30075841|pipeline II=54[2] |
|        loop i (cnn-krnl.cpp:25)    |118|                -|       -|-                 |
|            loop j (cnn-krnl.cpp:28)|118|                -|       -|-                 |
|    auto memory burst for 'A'(write)|   |    1769 (  0.0%)|    1769|cache size=113264B|
+------------------------------------+---+-----------------+--------+------------------+
""",
    "covariance_kernel.c": """
+---------------------------------------+---+----------------+-------+-----------------+
|               Hierarchy               |TC |       AC       |  CPC  |     Detail      |
+---------------------------------------+---+----------------+-------+-----------------+
|kernel_covariance (cnn-krnl.cpp:4)     |   |3554314 (100.0%)|3554314|-                |
|    auto memory burst for 'mean'(read) |   |     10 (  0.0%)|     10|cache size=640B  |
|    auto memory burst for 'data'(read) |   |   1000 (  0.0%)|   1000|cache size=64000B|
|    loop j (cnn-krnl.cpp:16)           | 80|  43120 (  1.2%)|  43120|-                |
|        loop i (cnn-krnl.cpp:20)       |100|  40160 (  1.1%)|    502|pipeline II=5    |
|    loop i (cnn-krnl.cpp:31)           |100|   8007 (  0.2%)|   8007|pipeline II=1    |
|        loop j (cnn-krnl.cpp:34)       | 80|               -|      -|flattened        |
|    auto memory burst for 'cov'(read)  |   |    800 (  0.0%)|    800|cache size=51200B|
|    loop i (cnn-krnl.cpp:44)           | 80|3500960 ( 98.5%)|3500960|-                |
|        loop j (cnn-krnl.cpp:47)       | 80|3500800 ( 98.5%)|  43760|-                |
|            loop k (cnn-krnl.cpp:51)   |100|3251200 ( 91.5%)|    508|pipeline II=5    |
|    auto memory burst for 'cov'(write) |   |    800 (  0.0%)|    800|cache size=51200B|
|    auto memory burst for 'data'(write)|   |   1000 (  0.0%)|   1000|cache size=64000B|
|    auto memory burst for 'mean'(write)|   |     10 (  0.0%)|     10|cache size=640B  |
+---------------------------------------+---+----------------+-------+-----------------+
""",
    "correlation_kernel.c": """
====================================================================================
Performance Estimate (TC: Trip Count, AC: Accumulated Cycles, CPC: Cycles Per Call)
====================================================================================

+-----------------------------------------+---+----------------+-------+-----------------+
|                Hierarchy                |TC |       AC       |  CPC  |     Detail      |
+-----------------------------------------+---+----------------+-------+-----------------+
|kernel_correlation (cnn-krnl.cpp:5)      |   |3323669 (100.0%)|3323669|-                |
|    auto memory burst for 'stddev'(read) |   |     10 (  0.0%)|     10|cache size=640B  |
|    auto memory burst for 'data'(read)   |   |   1000 (  0.0%)|   1000|cache size=64000B|
|    auto memory burst for 'corr'(read)   |   |   6400 (  0.2%)|   6400|cache size=51200B|
|    auto memory burst for 'mean'(read)   |   |     10 (  0.0%)|     10|cache size=640B  |
|    loop j (cnn-krnl.cpp:17)             | 80|  43120 (  1.3%)|  43120|-                |
|        loop i (cnn-krnl.cpp:21)         |100|  40160 (  1.2%)|    502|pipeline II=5    |
|    loop j (cnn-krnl.cpp:32)             | 80|  51360 (  1.5%)|  51360|-                |
|        loop i (cnn-krnl.cpp:36)         |100|  46400 (  1.4%)|    580|pipeline II=5    |
|    loop i (cnn-krnl.cpp:53)             |100|   8038 (  0.2%)|   8038|pipeline II=1    |
|        loop j (cnn-krnl.cpp:56)         | 80|               -|      -|flattened        |
|    loop i (cnn-krnl.cpp:68)             | 79|3208032 ( 96.5%)|3208032|-                |
|        loop j (cnn-krnl.cpp:72)         | 79|3207874 ( 96.5%)|  40606|-                |
|            loop k (cnn-krnl.cpp:76)     |100|3170428 ( 95.4%)|    508|pipeline II=5    |
|    auto memory burst for 'corr'(write)  |   |   6400 (  0.2%)|   6400|cache size=51200B|
|    auto memory burst for 'stddev'(write)|   |     10 (  0.0%)|     10|cache size=640B  |
|    auto memory burst for 'mean'(write)  |   |     10 (  0.0%)|     10|cache size=640B  |
|    auto memory burst for 'data'(write)  |   |   1000 (  0.0%)|   1000|cache size=64000B|
+-----------------------------------------+---+----------------+-------+-----------------+
""",
}
