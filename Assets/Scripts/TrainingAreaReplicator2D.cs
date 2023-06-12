using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class TrainingAreaReplicator2D : MonoBehaviour
{
    /// <summary>
    /// The base training area to be replicated.
    /// </summary>
    public GameObject baseArea;

    /// <summary>
    /// The number of training areas to replicate.
    /// </summary>
    public int numAreas = 1;

    /// <summary>
    /// The separation between each training area.
    /// </summary>
    public Vector2 separation = new Vector2(100f, 100f);

    Vector2 m_GridSize = new Vector2(1, 1);
    int m_areaCount = 0;
    string m_TrainingAreaName;

    /// <summary>
    /// The size of the computed grid to pack the training areas into.
    /// </summary>
    public Vector3 GridSize => m_GridSize;

    /// <summary>
    /// The name of the training area.
    /// </summary>
    public string TrainingAreaName => m_TrainingAreaName;

    /// <summary>
    /// Called before the simulation begins to computed the grid size for distributing
    /// the replicated training areas and set the area name.
    /// </summary>
    public void Awake()
    {
        // Computes the Grid Size on Awake
        ComputeGridSize();
        // Sets the TrainingArea name to the name of the base area.
        m_TrainingAreaName = baseArea.name;
    }

    /// <summary>
    /// Called after Awake and before the simulation begins and adds the training areas before
    /// the Academy begins.
    /// </summary>
    public void OnEnable()
    {
        // Adds the training are replicas during OnEnable to ensure they are added before the Academy begins its work.
        AddEnvironments();
    }

    /// <summary>
    /// Computes the Grid Size for replicating the training area.
    /// </summary>
    void ComputeGridSize()
    {
        // check if running inference, if so, use the num areas set through the component,
        // otherwise, pull it from the academy
        //if (Academy.Instance.IsCommunicatorOn)
        //    numAreas = Academy.Instance.NumAreas;

        var rootNumAreas = Mathf.Pow(numAreas, 1.0f / 2.0f);
        m_GridSize.x = Mathf.CeilToInt(rootNumAreas);
        m_GridSize.y = Mathf.CeilToInt(rootNumAreas);
    }

    /// <summary>
    /// Adds replicas of the training area to the scene.
    /// </summary>
    /// <exception cref="UnityAgentsException"></exception>
    void AddEnvironments()
    {
        if (numAreas > m_GridSize.x * m_GridSize.y)
        {
            throw new UnityAgentsException("The number of training areas that you have specified exceeds the size of the grid.");
        }

        for (int y = 0; y < m_GridSize.y; y++)
        {
            for (int x = 0; x < m_GridSize.x; x++)
            {
                if (m_areaCount == 0)
                {
                    // Skip this first area since it already exists.
                    m_areaCount = 1;
                }
                else if (m_areaCount < numAreas)
                {
                    m_areaCount++;
                    var area = Instantiate(baseArea, new Vector3(x * separation.x, y * separation.y, baseArea.transform.position.z), Quaternion.identity);
                    area.name = m_TrainingAreaName;
                }
            }
        }
    }
}
