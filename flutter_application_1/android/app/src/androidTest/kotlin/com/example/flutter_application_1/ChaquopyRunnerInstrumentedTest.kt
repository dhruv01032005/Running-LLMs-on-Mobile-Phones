package com.example.flutter_application_1

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ChaquopyRunnerInstrumentedTest {
    @Test
    fun runGeneratedPythonCode() {
        val generatedCode =
            """
            def true_code():
                import numpy as np
                import pandas as pd
                import pandas._libs.internals as pd_internals
                import matplotlib
                import seaborn as sns

                numbers = np.array([12, 18, 30])
                answer = int(pd.Series(numbers).sum())
                print(pd.__version__)
                print(hasattr(pd_internals, "_unpickle_block"))
                print(answer)

            true_code()
            """.trimIndent()

        Log.i(TAG, "Generated code:\n$generatedCode")
        println("Generated code:\n$generatedCode")

        val context = InstrumentationRegistry.getInstrumentation().targetContext
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }

        val module = Python.getInstance().getModule("chaquopy_runner")
        val pyResult = module.callAttr("run_code", generatedCode)
        val valuesByKey = mutableMapOf<String, String>()
        for ((keyObj, valueObj) in pyResult.asMap()) {
            valuesByKey[keyObj.toString()] = valueObj.toString()
        }

        val okValue = valuesByKey["ok"] ?: "False"
        val ok = okValue.equals("True", ignoreCase = true) || okValue.equals("true", ignoreCase = true)
        val stdout = valuesByKey["stdout"] ?: ""
        val stderr = valuesByKey["stderr"] ?: ""
        val traceback = valuesByKey["traceback"] ?: ""

        val answerText =
            if (ok) {
                when {
                    stdout.isNotBlank() -> stdout.trimEnd()
                    stderr.isNotBlank() -> stderr.trimEnd()
                    else -> "(Code executed with no output)"
                }
            } else {
                when {
                    traceback.isNotBlank() -> traceback.trimEnd()
                    stderr.isNotBlank() -> stderr.trimEnd()
                    else -> "Execution failed with no traceback"
                }
            }

        Log.i(TAG, "Chaquopy result:\n$answerText")
        println("Chaquopy result:\n$answerText")

        assertTrue("Chaquopy execution failed: $answerText", ok)
    }

    private companion object {
        const val TAG = "ChaquopyRunnerTest"
    }
}
