import axios from 'axios'

// API Base URL - connects to localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout for long operations
})

// Request interceptor for adding auth tokens
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMessage = error.response?.data?.detail || error.message || 'An error occurred'
    
    // Create standardized error object
    const standardError = new Error(errorMessage)
    standardError.status = error.response?.status
    standardError.data = error.response?.data
    standardError.isNetworkError = !error.response
    standardError.isTimeout = error.code === 'ECONNABORTED'
    
    return Promise.reject(standardError)
  }
)

// ============ Health API ============
export const healthApi = {
  check: async () => {
    const response = await api.get('/health')
    return response.data
  },
}

// ============ Jobs API ============
export const jobsApi = {
  getJobs: async () => {
    const response = await api.get('/api/jobs')
    return response.data
  },

  getJobById: async (jobId) => {
    const response = await api.get(`/api/jobs/${jobId}`)
    return response.data
  },

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

  deleteJob: async (jobId) => {
    const response = await api.delete(`/api/jobs/${jobId}`)
    return response.data
  },
}

// ============ Candidates API ============
export const candidatesApi = {
  addCandidate: async (jobId, candidateData) => {
    const response = await api.post(`/api/jobs/${jobId}/candidates`, {
      resume_text: candidateData.resume_text,
      resume_format: candidateData.resume_format || 'txt',
      source: candidateData.source || 'direct_application',
    })
    return response.data
  },

  getCandidates: async (jobId) => {
    const response = await api.get(`/api/jobs/${jobId}/candidates`)
    return response.data
  },

  getCandidateDetails: async (pipelineId, candidateId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/candidate/${candidateId}`)
    return response.data
  },
}

// ============ Pipeline API ============
export const pipelineApi = {
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

  runPipeline: async (pipelineId) => {
    const response = await api.post(`/api/pipelines/${pipelineId}/run`)
    return response.data
  },

  getPipeline: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}`)
    return response.data
  },

  // Get pipeline status for polling
  getStatus: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/status`)
    return response.data
  },

  getAuditLog: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/audit`)
    return response.data
  },

  getResults: async (pipelineId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/results`)
    return response.data
  },

  // List all pipelines
  listPipelines: async () => {
    const response = await api.get('/api/pipelines')
    return response.data
  },
}

// ============ Test API ============
export const testApi = {
  getQuestions: async (pipelineId, candidateId) => {
    const response = await api.get(`/api/pipelines/${pipelineId}/test`, {
      params: { candidate_id: candidateId },
    })
    return response.data
  },

  submitTest: async (pipelineId, candidateId, responses) => {
    const response = await api.post(`/api/pipelines/${pipelineId}/test/submit`, {
      candidate_id: candidateId,
      pipeline_id: pipelineId,
      responses: responses, // [{question_id, selected_option, time_seconds}]
    })
    return response.data
  },
}

// ============ Review API ============
export const reviewApi = {
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

// ============ Bias Audit API ============
export const biasApi = {
  getAuditResults: async (pipelineId) => {
    const results = await pipelineApi.getResults(pipelineId)
    return results.bias_audit || {}
  },
}

// ============ Polling Service ============
export class PipelinePoller {
  constructor(pipelineId, callbacks = {}) {
    this.pipelineId = pipelineId
    this.interval = null
    this.pollRate = 2000 // 2 seconds
    this.callbacks = {
      onStatusChange: callbacks.onStatusChange || (() => {}),
      onAwaitingReview: callbacks.onAwaitingReview || (() => {}),
      onCompleted: callbacks.onCompleted || (() => {}),
      onFailed: callbacks.onFailed || (() => {}),
      onError: callbacks.onError || (() => {}),
    }
    this.lastStatus = null
  }

  start() {
    this.stop() // Clear any existing interval
    this.poll() // Initial poll
    this.interval = setInterval(() => this.poll(), this.pollRate)
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval)
      this.interval = null
    }
  }

  async poll() {
    try {
      const data = await pipelineApi.getPipeline(this.pipelineId)
      const status = data.state?.current_stage || data.current_stage

      // Only trigger callbacks on status change
      if (status !== this.lastStatus) {
        this.lastStatus = status
        this.callbacks.onStatusChange(status, data)

        switch (status) {
          case 'awaiting_human_review':
            this.callbacks.onAwaitingReview(data)
            break
          case 'completed':
            this.callbacks.onCompleted(data)
            this.stop()
            break
          case 'failed':
            this.callbacks.onFailed(data)
            this.stop()
            break
        }
      }
    } catch (error) {
      this.callbacks.onError(error)
    }
  }

  setPollRate(ms) {
    this.pollRate = ms
    if (this.interval) {
      this.start() // Restart with new rate
    }
  }
}

// ============ Combined API Export ============
export default {
  health: healthApi,
  jobs: jobsApi,
  candidates: candidatesApi,
  pipeline: pipelineApi,
  test: testApi,
  review: reviewApi,
  bias: biasApi,
  PipelinePoller,
}
