using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace MBaske
{
    public class DroneAgent : Agent
    {
        [Header("Mode Settings")]
        public bool isTraining = true;

        [Header("Env Settings")]
        public Transform targetTransform;
        public Vector3 boundsSize = Vector3.one * 100f;
        public Objective objective;

        [SerializeField]
        private Multicopter multicopter = null;
        private Bounds bounds;
        private Resetter resetter;
        private Vector3 targetPos {
            get {
                return isTraining ? objective.currentTargetGlobal : targetTransform.position;
            }
        }

        private float prevDistToTarget;

        [System.Serializable]
        public class Objective {
            public int targetCount { get; private set; }
            public Vector3 currentTarget { get; private set; }
            public Vector3 currentTargetGlobal { get; private set; }
            [Range(0.05f, 1f)]
            public float targetRadius = 0.2f;
            public float timeIn { get; private set; }
            public float timeOut { get; private set; }
            public float points;

            public void Reset()
            {
                targetCount = 0;
                timeIn = 0f;
                timeOut = 0f;
                points = 0f;
                currentTarget = Vector3.zero;
            }

            public void NextTarget(Bounds bounds){
                targetCount++;
                currentTarget = new Vector3(
                    Random.Range(-3f, 3f),
                    Random.Range(-3f, 3f),
                    Random.Range(-3f, 3f)
                );
                currentTargetGlobal = currentTarget + bounds.center;
                timeIn = 0f;
                timeOut = 0f;
            }

            public void AddTimeIn(float dt){
                timeIn += dt;
            }

            public void AddTimeOut(float dt)
            {
                timeIn = 0f;
                timeOut += dt;
            }
        }

        public override void Initialize()
        {
            multicopter = GetComponentInChildren<Multicopter>();
            multicopter.Initialize();

            bounds = new Bounds(transform.position, boundsSize);
            resetter = new Resetter(transform);
        }

        public override void OnEpisodeBegin()
        {
            Multicopter multicopterChild = GetComponentInChildren<Multicopter>();
            if (multicopterChild != multicopter){
                multicopter = multicopterChild;
                multicopter.Initialize();
            }
            objective.NextTarget(bounds);
            prevDistToTarget = Vector3.Distance(objective.currentTargetGlobal, multicopter.Frame.position);
            resetter.Reset();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(multicopter.Inclination);
            sensor.AddObservation(Normalization.Sigmoid(
                multicopter.LocalizeVector(multicopter.Rigidbody.velocity), 0.25f));
            sensor.AddObservation(Normalization.Sigmoid(
                multicopter.LocalizeVector(multicopter.Rigidbody.angularVelocity)));
            
            foreach (var rotor in multicopter.Rotors)
            {
                sensor.AddObservation(rotor.CurrentThrust);
            }

            float maxDistance = 30f;
            Vector3 toTarget = targetPos - multicopter.Frame.position;
            float distToTarget = toTarget.magnitude;
            toTarget /= Mathf.Max(maxDistance, distToTarget);

            sensor.AddObservation(toTarget);

            sensor.AddObservation(multicopter.Frame.position - transform.position);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            /*bool cond = transform.position.magnitude < 0.1f;
            if (cond)
            {
                Debug.Log(actionBuffers.ContinuousActions.Array[0]);
                Debug.Log(actionBuffers.ContinuousActions.Array[1]);
                Debug.Log(actionBuffers.ContinuousActions.Array[2]);
                Debug.Log(actionBuffers.ContinuousActions.Array[3]);
                Debug.Log(transform.position);
            }*/
            multicopter.UpdateThrust(actionBuffers.ContinuousActions.Array);

            if (!bounds.Contains(multicopter.Frame.position))
            {
                resetter.Reset();
                return;
            }

            // Fitness stuffs
            float fitness = 0f;
            float distToTarget = Vector3.Distance(objective.currentTargetGlobal, multicopter.Frame.position);
            float distanceGain = prevDistToTarget - distToTarget;

            //fitness += 1.0f / (1.0f + distToTarget);
            fitness += distanceGain;

            // We don't want weirdos
            float scoreFactor = Mathf.Pow(multicopter.Frame.up.y, 2f);
            const float target_time = 0.5f;
            if (distToTarget < objective.targetRadius)
            {
                objective.AddTimeIn(Time.fixedDeltaTime);
                if (objective.timeIn > target_time)
                {
                    fitness += scoreFactor * objective.points / (1.0f + objective.timeOut);
                    objective.NextTarget(bounds);
                    objective.points = Vector3.Distance(objective.currentTargetGlobal, multicopter.Frame.position);
                }
            }
            else
            {
                objective.AddTimeOut(Time.fixedDeltaTime);
            }

            AddReward(fitness);
            AddReward(0.1f * multicopter.Frame.up.y);
            //AddReward(multicopter.Rigidbody.velocity.magnitude * -0.2f);
            AddReward(multicopter.Rigidbody.angularVelocity.magnitude * -0.1f);

            prevDistToTarget = distToTarget;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActionsOut = actionsOut.ContinuousActions;
            continuousActionsOut[0] = 0f;
            continuousActionsOut[1] = 0f;
            continuousActionsOut[2] = 0f;
            continuousActionsOut[3] = 0f;
        }

        public void OnDrawGizmos(){
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(transform.position, bounds.size);
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(targetPos, objective.targetRadius);
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(multicopter.Frame.position, objective.targetRadius);
        }
    }
}

