import axios from 'axios'

// API Base URL - defaults to local backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for long operations
})

// Request interceptor for adding auth tokens, etc.
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.error('Unauthorized access')
    }
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout')
    }
    return Promise.reject(error)
  }
)

// Health check
export const healthApi = {
  check: async () => {
    const response = await api.get('/health')
    return response.data
  },
}

// Jobs API endpoints
export const jobsApi = {
  // Get all jobs
  getJobs: async () => {
    const response = await api.get('/api/jobs')
    return response.data
  },

  // Get job by ID
  getJobById: async (jobId) => {
    const response = await api.get(`/api/jobs/${jobId}`)
    return response.data
  },

  // Create new job
  createJob: async (jobData) => {
    const response = await api.post('/api/jobs', {
      title: jobData.title,
      company: jobData.company || 'Anonymous Company',
      department: jobData.department || '',
      raw_description: jobData.raw_description || jobData.description,
      location: jobData.location || 'Remote',
      employment_type: jobData.employment_type || 'full_time',
      experience_years_min: jobData.experience_years_min || 0,
      experience_years_max: jobData.experience_years_max || 20,
    })
    return response.data
  },
}

// Candidates API endpoints
export const candidatesApi = {
  // Add a candidate to a job (text-based)
  addCandidate: async (jobId, candidateData) => {
    const response = await api.post(`/api/jobs/${jobId}/candidates`, {
      resume_text: candidateData.resume_text,
      resume_format: candidateData.resume_format || 'txt',
      source: candidateData.source || 'direct_application',
    })
    return response.data
  },

  // Upload a PDF resume for a candidate
  uploadResume: async (jobId, file, source = 'pdf_upload', onProgress = null) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source', source)

    const response = await api.post(`/api/jobs/${jobId}/candidates/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 second timeout for uploads
      onUploadProgress: onProgress ? (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onProgress(percentCompleted)
      } : undefined,
    })
    return response.data
  },

  // Get all candidates for a job
  getCandidates: async (jobId) => {
    const response = await api.get(`/api/jobs/${jobId}/candidates`)
    return response.data
  },

  // Get candidate details from pipeline
  getCandidateDetails: async (pipelineId, candidateId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/candidate/${candidateId}`)
    return response.data
  },
}

// Pipeline API endpoints
export const pipelineApi = {
  // Create a new pipeline
  createPipeline: async (jobId, candidateIds = [], config = {}) => {
    const response = await api.post('/api/pipelines', {
      job_id: jobId,
      candidate_ids: candidateIds,
      config: {
        shortlist_threshold: config.shortlist_threshold || 0.7,
        test_questions: config.test_questions || 10,
        test_passing_score: config.test_passing_score || 0.6,
        top_k_candidates: config.top_k_candidates || 10,
        ranking_weights: config.ranking_weights || { resume: 0.5, test: 0.5 },
      },
    })
    return response.data
  },

  // Run a pipeline
  runPipeline: async (pipelineId) => {
    const response = await api.post(`/api/pipelines/${pipelineId}/run`)
    return response.data
  },

  // Get pipeline status
  getPipeline: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}`)
    return response.data
  },

  // Get pipeline audit log
  getAuditLog: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/audit`)
    return response.data
  },

  // Get pipeline results
  getResults: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/results`)
    return response.data
  },
}

// Test API endpoints
export const testApi = {
  // Get test questions for a candidate
  getQuestions: async (pipelineId, candidateId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/test`, {
      params: { candidate_id: candidateId },
    })
    return response.data
  },

  // Submit test responses
  submitTest: async (pipelineId, candidateId, responses) => {
    const response = await api.post(`/api/pipelines/${pipelineId}/test/submit`, {
      candidate_id: candidateId,
      pipeline_id: pipelineId,
      responses: responses, // [{question_id, selected_option, time_seconds}]
    })
    return response.data
  },
}

// Human Review API endpoints
export const reviewApi = {
  // Submit human review decision
  submitReview: async (pipelineId, approved, notes = '', reviewer = 'anonymous') => {
    const response = await api.post(`/api/pipelines/${pipelineId}/review`, {
      pipeline_id: pipelineId,
      approved: approved,
      notes: notes,
      reviewer: reviewer,
    })
    return response.data
  },
}

// Bias Audit API endpoints (via pipeline)
export const biasApi = {
  // Get audit results from pipeline results
  getAuditResults: async (pipelineId) => {
    const results = await pipelineApi.getResults(pipelineId)
    return results.bias_audit || {}
  },
}

// Combined API for easier imports
export default {
  health: healthApi,
  jobs: jobsApi,
  candidates: candidatesApi,
  pipeline: pipelineApi,
  test: testApi,
  review: reviewApi,
  bias: biasApi,
}
