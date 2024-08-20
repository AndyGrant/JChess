/*
  Ethereal is a UCI chess playing engine authored by Andrew Grant.
  <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>

  Ethereal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Ethereal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef TUNE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitboards.h"
#include "board.h"
#include "evaluate.h"
#include "history.h"
#include "move.h"
#include "search.h"
#include "tuner.h"
#include "thread.h"
#include "transposition.h"
#include "types.h"
#include "uci.h"
#include "zobrist.h"

// Internal Memory Managment
TTuple* TupleStack;
int TupleStackSize = STACKSIZE;

// Tap into evaluate()
extern EvalTrace T, EmptyTrace;

extern const int PawnValue;
extern const int KnightValue;
extern const int BishopValue;
extern const int RookValue;
extern const int QueenValue;
extern const int PawnPSQT[64];
extern const int KnightPSQT[64];
extern const int BishopPSQT[64];
extern const int RookPSQT[64];
extern const int QueenPSQT[64];
extern const int KingPSQT[64];
extern const int PawnCandidatePasser[2][8];
extern const int PawnIsolated[8];
extern const int PawnStacked[2][8];
extern const int PawnBackwards[2][8];
extern const int PawnConnected32[32];
extern const int KnightOutpost[2][2];
extern const int KnightBehindPawn;
extern const int KnightInSiberia[4];
extern const int KnightMobility[9];
extern const int BishopPair;
extern const int BishopRammedPawns;
extern const int BishopOutpost[2][2];
extern const int BishopBehindPawn;
extern const int BishopLongDiagonal;
extern const int BishopMobility[14];
extern const int RookFile[2];
extern const int RookOnSeventh;
extern const int RookMobility[15];
extern const int QueenRelativePin;
extern const int QueenMobility[28];
extern const int KingDefenders[12];
extern const int KingPawnFileProximity[8];
extern const int KingShelter[2][8][8];
extern const int KingStorm[2][4][8];
extern const int SafetyKnightWeight;
extern const int SafetyBishopWeight;
extern const int SafetyRookWeight;
extern const int SafetyQueenWeight;
extern const int SafetyAttackValue;
extern const int SafetyWeakSquares;
extern const int SafetyNoEnemyQueens;
extern const int SafetySafeQueenCheck;
extern const int SafetySafeRookCheck;
extern const int SafetySafeBishopCheck;
extern const int SafetySafeKnightCheck;
extern const int SafetyAdjustment;
extern const int SafetyShelter[2][8];
extern const int SafetyStorm[2][8];
extern const int PassedPawn[2][2][8];
extern const int PassedFriendlyDistance[8];
extern const int PassedEnemyDistance[8];
extern const int PassedSafePromotionPath;
extern const int ThreatWeakPawn;
extern const int ThreatMinorAttackedByPawn;
extern const int ThreatMinorAttackedByMinor;
extern const int ThreatMinorAttackedByMajor;
extern const int ThreatRookAttackedByLesser;
extern const int ThreatMinorAttackedByKing;
extern const int ThreatRookAttackedByKing;
extern const int ThreatQueenAttackedByOne;
extern const int ThreatOverloadedPieces;
extern const int ThreatByPawnPush;
extern const int SpaceRestrictPiece;
extern const int SpaceRestrictEmpty;
extern const int SpaceCenterControl;
extern const int ClosednessKnightAdjustment[9];
extern const int ClosednessRookAdjustment[9];
extern const int ComplexityTotalPawns;
extern const int ComplexityPawnFlanks;
extern const int ComplexityPawnEndgame;
extern const int ComplexityAdjustment;
extern const int Tempo;


void runTuner() {

    TEntry *entries;
    TArray methods = {0};
    TVector params = {0}, cparams = {0}, adagrad = {0};
    Thread *thread = createThreadPool(1);
    double K, error, rate = LRRATE;

    const int tentryMB = (int)(NPOSITIONS * sizeof(TEntry) / (1 << 20));
    const int ttupleMB = (int)(STACKSIZE  * sizeof(TTuple) / (1 << 20));

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Tuner will be tuning 2x%d Terms\n", NTERMS);
    printf("Allocating Memory for Tuner Entries [%dMB]\n", tentryMB);
    printf("Allocating Memory for Tuner Tuple Stack [%dMB]\n", ttupleMB);
    printf("Saving the current value for each Term as a starting point\n");
    printf("Marking each Term based on method { NORMAL, SAFETY, COMPLEXITY }\n\n");

    entries    = calloc(NPOSITIONS, sizeof(TEntry));
    TupleStack = calloc(STACKSIZE , sizeof(TTuple));

    initCurrentParameters(cparams);
    initMethodManager(methods);
    initTunerEntries(entries, thread, methods);
    K = computeOptimalK(entries);

    for (int epoch = 0; epoch < MAXEPOCHS; epoch++) {

        for (int batch = 0; batch < NPOSITIONS / BATCHSIZE; batch++) {

            TVector gradient = {0};
            computeGradient(entries, gradient, params, methods, K, batch);

            for (int i = 0; i < NTERMS; i++) {
                adagrad[i][MG] += pow((K / 200.0) * gradient[i][MG] / BATCHSIZE, 2.0);
                adagrad[i][EG] += pow((K / 200.0) * gradient[i][EG] / BATCHSIZE, 2.0);
                params[i][MG] += (K / 200.0) * (gradient[i][MG] / BATCHSIZE) * (rate / sqrt(1e-8 + adagrad[i][MG]));
                params[i][EG] += (K / 200.0) * (gradient[i][EG] / BATCHSIZE) * (rate / sqrt(1e-8 + adagrad[i][EG]));
            }
        }

        error = tunedEvaluationErrors(entries, params, methods, K);
        if (epoch && epoch % LRSTEPRATE == 0) rate = rate / LRDROPRATE;
        if (epoch % REPORTING == 0) printParameters(params, cparams);

        printf("\rEpoch [%d] Error = [%.9f], Rate = [%g]", epoch, error, rate);
    }
}

void initCurrentParameters(TVector cparams) {

    int i = 0; // EXECUTE_ON_TERMS will update i accordingly

    EXECUTE_ON_TERMS(INIT_PARAM);

    if (i != NTERMS){
        printf("Error in initCurrentParameters(): i = %d ; NTERMS = %d\n", i, NTERMS);
        exit(EXIT_FAILURE);
    }
}

void initMethodManager(TArray methods) {

    int i = 0; // EXECUTE_ON_TERMS will update i accordingly

    EXECUTE_ON_TERMS(INIT_METHOD);

    if (i != NTERMS){
        printf("Error in initMethodManager(): i = %d ; NTERMS = %d\n", i, NTERMS);
        exit(EXIT_FAILURE);
    }
}

void initCoefficients(TVector coeffs) {

    int i = 0; // EXECUTE_ON_TERMS will update i accordingly

    EXECUTE_ON_TERMS(INIT_COEFF);

    if (i != NTERMS){
        printf("Error in initCoefficients(): i = %d ; NTERMS = %d\n", i, NTERMS);
        exit(EXIT_FAILURE);
    }
}

void initTunerEntries(TEntry *entries, Thread *thread, TArray methods) {

    char line[256];
    FILE *fin = fopen("FENS", "r");

    for (int i = 0; i < NPOSITIONS; i++) {

        if (fgets(line, 256, fin) == NULL)
            exit(EXIT_FAILURE);

        // Find the result { W, L, D } => { 1.0, 0.0, 0.5 }
        if      (strstr(line, "[1.0]")) entries[i].result = 1.0;
        else if (strstr(line, "[0.0]")) entries[i].result = 0.0;
        else if (strstr(line, "[0.5]")) entries[i].result = 0.5;
        else    {printf("Cannot Parse %s\n", line); exit(EXIT_FAILURE);}

        // Set the board with the current FEN
        boardFromFEN(&thread->board, line, 0);

        // Defer the setup to another function
        initTunerEntry(&entries[i], thread, &thread->board, methods);

        // Occasional reporting for total completion
        if ((i + 1) % 10000 == 0 || i == NPOSITIONS - 1)
            printf("\rSetting up Entries from FENs [%8d of %8d]", i + 1, NPOSITIONS);
    }

    fclose(fin);
}

void initTunerEntry(TEntry *entry, Thread *thread, Board *board, TArray methods) {

    // Use the same phase calculation as evaluate()
    int phase = 4 * popcount(board->pieces[QUEEN ])
              + 2 * popcount(board->pieces[ROOK  ])
              + 1 * popcount(board->pieces[BISHOP])
              + 1 * popcount(board->pieces[KNIGHT]);

    // Save time by computing phase scalars now
    entry->pfactors[MG] = 0 + phase / 24.0;
    entry->pfactors[EG] = 1 - phase / 24.0;
    entry->phase = phase;

    // Save a white POV static evaluation
    TVector coeffs; T = EmptyTrace;
    entry->seval = evaluateBoard(thread, board);
    if (board->turn == BLACK) entry->seval = -entry->seval;

    // evaluate() -> [[NTERMS][COLOUR_NB]]
    initCoefficients(coeffs);
    initTunerTuples(entry, coeffs, methods);

    // Save some of the evaluation modifiers
    entry->eval        = T.eval;
    entry->complexity  = T.complexity;
    entry->sfactor     = T.factor / (double) SCALE_NORMAL;
    entry->turn        = board->turn;

    // Save the Linear version of King Safety
    entry->safety[WHITE] = T.safety[WHITE];
    entry->safety[BLACK] = T.safety[BLACK];
}

void initTunerTuples(TEntry *entry, TVector coeffs, TArray methods) {

    int ttupleMB, length = 0, tidx = 0;

    // Sum up any actively used terms
    for (int i = 0; i < NTERMS; i++)
        length += (methods[i] == NORMAL &&  coeffs[i][WHITE] - coeffs[i][BLACK] != 0.0)
               || (methods[i] != NORMAL && (coeffs[i][WHITE] != 0.0 || coeffs[i][BLACK] != 0.0));

    // Allocate additional memory if needed
    if (length > TupleStackSize) {
        TupleStackSize = STACKSIZE;
        TupleStack = calloc(STACKSIZE, sizeof(TTuple));
        ttupleMB = STACKSIZE  * sizeof(TTuple) / (1 << 20);
        printf(" Allocating Tuner Tuples [%dMB]\n", ttupleMB);
    }

    // Claim part of the Tuple Stack
    entry->tuples   = TupleStack;
    entry->ntuples  = length;
    TupleStack     += length;
    TupleStackSize -= length;

    // Finally setup each of our TTuples
    for (int i = 0; i < NTERMS; i++)
        if (   (methods[i] == NORMAL &&  coeffs[i][WHITE] - coeffs[i][BLACK] != 0.0)
            || (methods[i] != NORMAL && (coeffs[i][WHITE] != 0.0 || coeffs[i][BLACK] != 0.0)))
            entry->tuples[tidx++] = (TTuple) { i, coeffs[i][WHITE], coeffs[i][BLACK] };
}


double computeOptimalK(TEntry *entries) {

    const double eps = pow(10, -KPRECISION);
    double low = -10, high = 10, mid;
    double E1, E2;

    printf("\n\nComputing optimal K\n");

    while (low + eps <= high) {
        mid = (low + high) / 2.0;
        E1 = staticEvaluationErrors(entries, mid);
        E2 = staticEvaluationErrors(entries, mid + eps);
        if (E1 < E2) {
            high = mid;
        } else {
            low = mid + eps;
        }
    }

    printf("K = [%.9f] E = [%.9f]\n", low, E1);
    printf("\n");

    return low;
}

double staticEvaluationErrors(TEntry *entries, double K) {

    double total = 0.0;

    #pragma omp parallel shared(total)
    {
        #pragma omp for schedule(static, NPOSITIONS / NPARTITIONS) reduction(+:total)
        for (int i = 0; i < NPOSITIONS; i++)
            total += pow(entries[i].result - sigmoid(K, entries[i].seval), 2);
    }

    return total / (double) NPOSITIONS;
}

double tunedEvaluationErrors(TEntry *entries, TVector params, TArray methods, double K) {

    double total = 0.0;

    #pragma omp parallel shared(total)
    {
        #pragma omp for schedule(static, NPOSITIONS / NPARTITIONS) reduction(+:total)
        for (int i = 0; i < NPOSITIONS; i++)
            total += pow(entries[i].result - sigmoid(K, linearEvaluation(&entries[i], params, methods, NULL)), 2);
    }

    return total / (double) NPOSITIONS;
}

double sigmoid(double K, double E) {
    return 1.0 / (1.0 + exp(-K * E / 400.0));
}


double linearEvaluation(TEntry *entry, TVector params, TArray methods, TGradientData *data) {

    double sign, mixed;
    double midgame, endgame, wsafety[2], bsafety[2];
    double normal[PHASE_NB], safety[PHASE_NB], complexity;
    double mg[METHOD_NB][COLOUR_NB] = {0}, eg[METHOD_NB][COLOUR_NB] = {0};

    // Save any modifications for MG or EG for each evaluation type
    for (int i = 0; i < entry->ntuples; i++) {
        int index = entry->tuples[i].index;
        mg[methods[index]][WHITE] += (double) entry->tuples[i].wcoeff * params[index][MG];
        mg[methods[index]][BLACK] += (double) entry->tuples[i].bcoeff * params[index][MG];
        eg[methods[index]][WHITE] += (double) entry->tuples[i].wcoeff * params[index][EG];
        eg[methods[index]][BLACK] += (double) entry->tuples[i].bcoeff * params[index][EG];
    }

    // Grab the original "normal" evaluations and add the modified parameters
    normal[MG] = (double) ScoreMG(entry->eval) + mg[NORMAL][WHITE] - mg[NORMAL][BLACK];
    normal[EG] = (double) ScoreEG(entry->eval) + eg[NORMAL][WHITE] - eg[NORMAL][BLACK];

    // Grab the original "safety" evaluations and add the modified parameters
    wsafety[MG] = (double) ScoreMG(entry->safety[WHITE]) + mg[SAFETY][WHITE];
    wsafety[EG] = (double) ScoreEG(entry->safety[WHITE]) + eg[SAFETY][WHITE];
    bsafety[MG] = (double) ScoreMG(entry->safety[BLACK]) + mg[SAFETY][BLACK];
    bsafety[EG] = (double) ScoreEG(entry->safety[BLACK]) + eg[SAFETY][BLACK];

    // Remove the original "safety" evaluation that was double counted into the "normal" evaluation
    normal[MG] -= MIN(0, -ScoreMG(entry->safety[WHITE]) * fabs(ScoreMG(entry->safety[WHITE])) / 720.0)
                - MIN(0, -ScoreMG(entry->safety[BLACK]) * fabs(ScoreMG(entry->safety[BLACK])) / 720.0);
    normal[EG] -= MIN(0, -ScoreEG(entry->safety[WHITE]) / 20.0) - MIN(0, -ScoreEG(entry->safety[BLACK]) / 20.0);

    // Compute the new, true "safety" evaluations for each side
    safety[MG] = MIN(0, -wsafety[MG] * fabs(-wsafety[MG]) / 720.0)
               - MIN(0, -bsafety[MG] * fabs(-bsafety[MG]) / 720.0);
    safety[EG] = MIN(0, -wsafety[EG] / 20.0) - MIN(0, -bsafety[EG] / 20.0);

    // Grab the original "complexity" evaluation and add the modified parameters
    complexity = (double) ScoreEG(entry->complexity) + eg[COMPLEXITY][WHITE];
    sign       = (normal[EG] + safety[EG] > 0.0) - (normal[EG] + safety[EG] < 0.0);

    // Save this information since we need it to compute the gradients
    if (data != NULL)
        *data = (TGradientData) {
            normal[EG] + safety[EG], complexity,
            wsafety[MG], bsafety[MG], wsafety[EG], bsafety[EG]
        };

    midgame = normal[MG] + safety[MG];
    endgame = normal[EG] + safety[EG] + sign * fmax(-fabs(normal[EG] + safety[EG]), complexity);

    mixed = (midgame * entry->phase
          +  endgame * (24.0 - entry->phase) * entry->sfactor) / 24.0;

    return mixed + (entry->turn == WHITE ? Tempo : -Tempo);
}

void computeGradient(TEntry *entries, TVector gradient, TVector params, TArray methods, double K, int batch) {

    #pragma omp parallel shared(gradient)
    {
        TVector local = {0};

        #pragma omp for schedule(static, BATCHSIZE / NPARTITIONS)
        for (int i = batch * BATCHSIZE; i < (batch + 1) * BATCHSIZE; i++)
            updateSingleGradient(&entries[i], local, params, methods, K);

        for (int i = 0; i < NTERMS; i++) {
            gradient[i][MG] += local[i][MG];
            gradient[i][EG] += local[i][EG];
        }
    }
}

void updateSingleGradient(TEntry *entry, TVector gradient, TVector params, TArray methods, double K) {

    TGradientData data;
    double E = linearEvaluation(entry, params, methods, &data);
    double S = sigmoid(K, E);
    double A = (entry->result - S) * S * (1 - S);

    double mgBase = A * entry->pfactors[MG];
    double egBase = A * entry->pfactors[EG];

    double complexitySign = (data.egeval > 0.0) - (data.egeval < 0.0);

    for (int i = 0; i < entry->ntuples; i++) {

        int index  = entry->tuples[i].index;
        int wcoeff = entry->tuples[i].wcoeff;
        int bcoeff = entry->tuples[i].bcoeff;

        if (methods[index] == NORMAL)
            gradient[index][MG] += mgBase * (wcoeff - bcoeff);

        if (methods[index] == NORMAL && (data.egeval == 0.0 || data.complexity >= -fabs(data.egeval)))
            gradient[index][EG] += egBase * (wcoeff - bcoeff) * entry->sfactor;

        if (methods[index] == COMPLEXITY && data.complexity >= -fabs(data.egeval))
            gradient[index][EG] += egBase * wcoeff * complexitySign * entry->sfactor;

        if (methods[index] == SAFETY)
            gradient[index][MG] += (mgBase / 360.0) * (fmax(data.bsafetymg, 0) * bcoeff - (fmax(data.wsafetymg, 0) * wcoeff));

        if (methods[index] == SAFETY && (data.egeval == 0.0 || data.complexity >= -fabs(data.egeval)))
            gradient[index][EG] += (egBase /  20.0) * ((data.bsafetyeg > 0.0) * bcoeff - (data.wsafetyeg > 0.0) * wcoeff);
    }
}


void printParameters(TVector params, TVector cparams) {

    TVector tparams;

    // Combine updated and current parameters
    for (int j = 0; j < NTERMS; j++) {
        tparams[j][MG] = round(params[j][MG] + cparams[j][MG]);
        tparams[j][EG] = round(params[j][EG] + cparams[j][EG]);
    }

    int i = 0; // EXECUTE_ON_TERMS will update i accordingly

    EXECUTE_ON_TERMS(PRINT);

    if (i != NTERMS) {
        printf("Error in printParameters(): i = %d ; NTERMS = %d\n", i, NTERMS);
        exit(EXIT_FAILURE);
    }
}

void print_0(char *name, TVector params, int i, char *S) {

    printf("const int %s%s = S(%4d,%4d);\n", name, S, (int) params[i][MG], (int) params[i][EG]);

}

void print_1(char *name, TVector params, int i, int A, char *S) {

    printf("const int %s%s = { ", name, S);

    if (A >= 3) {

        for (int a = 0; a < A; a++, i++) {
            if (a % 4 == 0) printf("\n    ");
            printf("S(%4d,%4d), ", (int) params[i][MG], (int) params[i][EG]);
        }

        printf("\n};\n\n");
    }

    else {

        for (int a = 0; a < A; a++, i++) {
            printf("S(%4d,%4d)", (int) params[i][MG], (int) params[i][EG]);
            if (a != A - 1) printf(", "); else printf(" };\n\n");
        }
    }

}

void print_2(char *name, TVector params, int i, int A, int B, char *S) {

    printf("const int %s%s = {\n", name, S);

    for (int a = 0; a < A; a++) {

        printf("   {");

        for (int b = 0; b < B; b++, i++) {
            if (b && b % 4 == 0) printf("\n    ");
            printf("S(%4d,%4d)", (int) params[i][MG], (int) params[i][EG]);
            printf("%s", b == B - 1 ? "" : ", ");
        }

        printf("},\n");
    }

    printf("};\n\n");

}

void print_3(char *name, TVector params, int i, int A, int B, int C, char *S) {

    printf("const int %s%s = {\n", name, S);

    for (int a = 0; a < A; a++) {

        for (int b = 0; b < B; b++) {

            printf("%s", b ? "   {" : "  {{");;

            for (int c = 0; c < C; c++, i++) {
                if (c &&  c % 4 == 0) printf("\n    ");
                printf("S(%4d,%4d)", (int) params[i][MG], (int) params[i][EG]);
                printf("%s", c == C - 1 ? "" : ", ");
            }

            printf("%s", b == B - 1 ? "}},\n" : "},\n");
        }

    }

    printf("};\n\n");
}

#endif
