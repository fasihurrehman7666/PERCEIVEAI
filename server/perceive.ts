import axios from "axios";

const PERCEIVE_API_URL = "http://127.0.0.1:8000";

export interface PerceiveGradeRequestQuestion {
  questionId: number;
  prompt: string;
  correctAnswer: string;
  studentAnswer: string;
}

export interface PerceiveGradeRequest {
  submissionId: string;
  quizType: "assigned" | "practice";
  studentId: number;
  quizId?: number;
  practiceQuizId?: number;
  subject?: string;
  questions: PerceiveGradeRequestQuestion[];
  metadata?: Record<string, unknown>;
}

export type PerceiveLetterGrade = "A" | "B" | "C" | "D" | "F";

export interface PerceiveQuestionGrade {
  questionId: number;
  letterGrade: PerceiveLetterGrade;
  feedback: string;
}

export interface PerceiveGradeResponse {
  submissionId: string;
  questionGrades: PerceiveQuestionGrade[];
  model: string;
  latencyMs?: number;
}

export async function gradeWithPerceive(request: PerceiveGradeRequest): Promise<PerceiveGradeResponse> {
  const start = Date.now();
  const questionGrades: PerceiveQuestionGrade[] = [];

  for (const question of request.questions) {
    try {
      // Check for empty answers first
      if (!question.studentAnswer || question.studentAnswer.trim() === "") {
        questionGrades.push({
          questionId: question.questionId,
          letterGrade: "F",
          feedback: "Empty answer submitted."
        });
        continue;
      }

      const response = await axios.post(`${PERCEIVE_API_URL}/grade`, {
        qid: question.questionId,
        question: question.prompt,
        answer: question.studentAnswer
      }, {
        timeout: 30000 // 30s timeout per question for BERT model processing
      });

      const letterGrade = response.data.grade as PerceiveLetterGrade;
      
      questionGrades.push({
        questionId: question.questionId,
        letterGrade: letterGrade,
        feedback: `Perceive AI assigned grade ${letterGrade} based on answer evaluation.`
      });
    } catch (error: any) {
      console.error(`Perceive AI grading failed for question ${question.questionId}:`, error.message);
      questionGrades.push({
        questionId: question.questionId,
        letterGrade: "C", // Fallback grade
        feedback: "Auto-grading temporarily unavailable, assigned default grade."
      });
    }
  }

  return {
    submissionId: request.submissionId,
    questionGrades,
    model: "Perceive AI BERT Model",
    latencyMs: Date.now() - start
  };
}
