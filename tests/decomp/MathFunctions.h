#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include "Common.h"

// ------------------------------------------------------------------------------------
namespace PBD
{
	class MathFunctions
	{
	public:
		static void jacobiRotate(Matrix3r &A,
			Matrix3r &R,
			int p,
			int q);

	public:
		static Real infNorm(const Matrix3r &A);
		static Real oneNorm(const Matrix3r &A);

		static void eigenDecomposition(const Matrix3r &A,
			Matrix3r &eigenVecs,
			Vector3r &eigenVals);

		static void polarDecomposition(const Matrix3r &A,
			Matrix3r &R,
			Matrix3r &U,
			Matrix3r &D);

		static void polarDecompositionStable(const Matrix3r &M,
			const Real tolerance,
			Matrix3r &R);

		static void svdWithInversionHandling(const Matrix3r &A,
			Vector3r &sigma,
			Matrix3r &U,
			Matrix3r &VT);

		static Real cotTheta(const Vector3r &v, const Vector3r &w);

		/** Computes the cross product matrix of a vector.
		 * @param  v		input vector
		 * @param  v_hat	resulting cross product matrix
		 */	
		static void crossProductMatrix(const Vector3r &v, Matrix3r &v_hat);
	};
}

#endif